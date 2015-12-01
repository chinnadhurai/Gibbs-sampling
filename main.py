__author__ = 'chinna'
import theano
from theano import tensor as T
import numpy as np
import matplotlib.pyplot as plt
from theano import shared
from theano import function
import scipy as sp
from scipy import signal
from PIL import Image

class gibbs_sampler_class:
    def __init__(self,config):
        print "Creating Gibbs sampler..."
        self.config = config
        np.random.seed(self.config['seed'])
        self.d = config['grid_size']
        self.X = np.zeros((self.d+2,self.d+2))
        p = config['p']
        d = self.d
        self.X[1:d+1,1:d+1] = np.random.choice(np.array([1,-1]),size=(d,d),p=[p,1-p])#2*np.random.binomial(1,config['p'],(self.d,self.d)) - 1
        self.row = 1
        self.col = 0
        print "Grid size :",self.d,'x',self.d
        print "initilization done..."

    # iterator ( row wise )
    def get_next(self):
        self.col += 1
        # end of matrix, start over
        if self.row == self.d and self.col > self.d:
            self.row = self.col = 1
        # end of row. go to next row
        if self.col > self.d:
            self.col  = 1
            self.row += 1
        return self.row, self.col

    def nbr_sum(self,X,i,j):
        return X[i,j+1] + X[i,j-1] + X[i+1,j] + X[i-1,j]

    def sample(self,i,j):
        X,t = self. X,self.config['theta']
        s = self.nbr_sum(X,i,j)
        p = np.exp(t*s)/(np.exp(t*s) + np.exp(-t*s))
        return np.random.choice(np.array([1,-1]),size=(1,),p=[p,1-p])#2*np.random.binomial(1,p) - 1

    def plot(self,file,X):
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        ax.set_title('colorMap')
        plt.pcolor(X,cmap='Greys')#'Reds')
        plt.savefig(file)
        print "plot saved as :",file

    def run(self):
        self.plot(self.config['name'] + "_before.jpeg",self.X[1:self.d+1,1:self.d+1])
        L_heatmap = []
        for k in range(self.config['iterations']+1):
            for m_t in range((self.d**2)):
                i,j = self.get_next()
                self.X[i,j] = self.sample(i,j)
            L_heatmap.append(self.X[1:self.d+1,1:self.d+1].flatten())
            if k % self.config['visualize_checkpt'] == 0:
                    print "#Iterations completed:",k
                    self.plot(self.config['name'] + "_itr_"+str(k)+".jpeg",self.X[1:self.d+1,1:self.d+1])
        self.plot("vanilla_gibbs_heat_map.jpeg",np.array(L_heatmap))
""" ================================================= """
""" ================================================= """
""" =========== Block-Gibbs sampling =============== """
""" ================================================= """
""" ================================================= """

class node:
    num_nodes = 0
    def __init__(self,theta,d=1):
        self.__class__.num_nodes += 1
        #print "creating node...",self.__class__.num_nodes
        self.node_id = self.__class__.num_nodes
        self.t = theta
        p = 0.5
        self.x = np.random.choice(np.array([1,-1]),size=(2,),p=[p,1-p])
        self.l = None
        self.r = None
        x = np.exp(self.t)
        self.bf = np.array([[x,1/x],[1/x,x]])

    def update_bf(self,a,b):
        t = self.t
        for i in range(2):
            for j in range(2):
                x1 = 2*i - 1
                x2 = 2*j - 1
                self.bf[i][j] *= np.exp(t*x1*a + t*x2*b)

    def normalize_bf(self):
        self.bf = self.bf/(self.bf.sum())

class block_gibbs_sampler_class:
    def __init__(self,config):
        self.config = config
        print "Initializing for belief prop..."
        np.random.seed(config['seed'])
        self.d       = config['num_nodes']
        self.n_m     = config['num_messages']
        self.k       = config['variable_cardinality']
        self.t       = config['theta']
        x = np.exp(self.t)
        self.phi = np.array([[x,1/x],[1/x,x]])
        self.nodes = []#np.array([node(t) for i in range(d)])
        self.root = [None]*2
        #self.msgs  = np.array([belief(k) for i in range(n_m)])

    def plot(self,file,X):
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        ax.set_title('colorMap')
        plt.pcolor(X,cmap='Greys')#'Reds')
        plt.savefig(file)
        print "plot saved as :",file

    def add_nodes_to_left(self,root,d):
        itr = root
        for i in range(d):
            itr.l = node(self.t)
            itr = itr.l
        return itr

    def add_nodes_to_right(self,root,d):
        itr = root
        for i in range(d):
            itr.r = node(self.t)
            itr = itr.r
        return itr

    #comb structure
    def create_clique_tree(self,d):
        root = node(self.t)
        self.add_nodes_to_left(root, d-2)
        itr = root
        for i in range(d):
            itr.r = node(self.t)
            itr = itr.r
            if i%2 != 0:
                self.add_nodes_to_left(itr,d-1)
        return root
    #update the phi / beliefs in the function
    #from the nbr samples
    def update_clique_tree(self,root,samples):
        assert len(samples) == self.d / 2
        # treat the first branch alone differently
        #  due to the assymetry in the clique tree
        tmp_itr = root.l

        for j in range(self.d - 3):
            tmp_itr.update_bf(samples[0][j],0)
            if j == self.d - 4:
                tmp_itr.update_bf(0,samples[0][j+1])
                tmp_itr.update_bf(0,samples[0][j+2])
            tmp_itr = tmp_itr.l

        sample_comb = 0
        itr = root.r
        for i in range(self.d-1):
            if i%2 == 0:
                itr.update_bf(0,samples[sample_comb][0])
            else:
                tmp_itr = itr.l
                for j in range(self.d - 2):
                    tmp_itr.update_bf(samples[sample_comb][j],0)
                    tmp_itr.update_bf(0,samples[sample_comb+1][j])
                    if j == self.d - 3:
                        tmp_itr.update_bf(0,samples[sample_comb+1][j+1])
                    tmp_itr = tmp_itr.l
                sample_comb += 1
            itr = itr.r

    def gen_message_down_bp(self,root,m):
        # include the incoming the message over the left item
        for i in range(2):
            for j in range(2):
                root.bf[i][j] *= m[i]
        # gen message by marglinizing the left item
        return root.bf.sum(axis=0)

    #preorder traversal
    def downward_msg_pass_util(self,root,m,p_type):
        if root == None:
            return None
        #print "visiting node id:",root.node_id

        #get message
        new_m = self.gen_message_down_bp(root,m)

        #pass the message to the children
        self.downward_msg_pass_util(root.l, new_m,0)
        self.downward_msg_pass_util(root.r, new_m,1)

    def downward_msg_pass(self,root):

        print "Triggering downward pass..."
        #Gen explicit msgs for the root due to the assymtry
        rmsg = root.bf.sum(axis=0)
        lmsg = root.bf.sum(axis=1)
        self.downward_msg_pass_util(root.r,rmsg,0)
        self.downward_msg_pass_util(root.l,lmsg,0)
        print "downward pass done..."

    def gen_message_up_bp(self,root,m):
        # include the incoming the message over the right item
        for i in range(2):
            for j in range(2):
                root.bf[i][j] *= m[j]
        # gen message by marglinizing the right item
        return root.bf.sum(axis=1)

    # post order traversal
    def upward_msg_pass_util(self,root):
        if root == None:
            return np.ones(2)
        lmsg = self.upward_msg_pass_util(root.l)
        rmsg = self.upward_msg_pass_util(root.r)
        #print "visiting node id :",root.node_id, lmsg
        new_m = self.gen_message_up_bp(root, lmsg * rmsg)
        return new_m

    def upward_msg_pass(self,root):

        #print "Triggering upward pass..."
        lmsg = self.upward_msg_pass_util(root.r)
        rmsg = self.upward_msg_pass_util(root.l)

        # dealing with root here due to the assymtry
        for i in range(2):
            for j in range(2):
                root.bf[i][j] *= lmsg[j]*rmsg[i]
        #print "upward pass done..."

    def get_sample_with(self,p):
        return np.random.choice(np.array([1,-1]),size=(1,),p=[p,1-p])

    def get_samples_util(self,root,p_x,s_x,L_samples):
        if root == None:
            return
        p_child_parent = root.bf[(s_x+1)/2,:] # choose the row based on s_x
        p_child_given_parent = p_child_parent / p_x[(s_x+1)/2]
        p_child_given_parent = p_child_given_parent/p_child_given_parent.sum()

        new_sample = self.get_sample_with(p_child_given_parent[0,1])
        L_samples.append(new_sample)
        #print "appending sample",new_sample, len(L_samples), root.node_id
        p_child = (root.bf.sum(axis=0)) #marginalize left item
        p_child = p_child/p_child.sum()

        self.get_samples_util(root.l, p_child, new_sample, L_samples)
        self.get_samples_util(root.r, p_child, new_sample, L_samples)


    def get_samples(self,root):
        #print "getting samples...."
        # get two samples from the root node and give one to each child
        L_samples = []
        p_x11 = root.bf.sum(axis=1)
        p_x11 = p_x11/p_x11.sum()
        s_x11 = self.get_sample_with(p_x11[1])

        p_x21 = root.bf.sum(axis=0)
        p_x21 = p_x21/p_x21.sum()
        s_x21 = self.get_sample_with(p_x21[1])
        L_samples.append(s_x11)
        L_samples.append(s_x21)
        self.get_samples_util(root.l,p_x21,s_x21,L_samples)
        self.get_samples_util(root.r,p_x11,s_x11,L_samples)
        assert len(L_samples) == (self.d**2)/2

        formatted_samples = np.ones((self.d/2, self.d-1))
        d = self.d
        for i in range(self.d/2):
            for j in range(d-1):
                formatted_samples[-i-1,j] = L_samples[i*d+j]
                if j == d-2:
                    formatted_samples[-i-1,j] = L_samples[i*d+d-1]
            #print formatted_samples[-i-1]

        #print "sampling done...",len(L_samples)
        return formatted_samples, L_samples

    def normlize_util(self,root):
        if root == None:
            return
        root.normalize_bf()
        self.normlize_util(root.l)
        self.normlize_util(root.r)

    def normalize(self,root):
        print "Normalizing beliefs..."
        self.normlize_util(root)

    def visualize(self,b,itr_num):
        d = self.d
        visual = np.zeros((d,d))
        for j in range(d):
            if j % 2 ==0:
                for i in range(d-1):
                    visual[i][j] = b[0][(d*j)/2 + i]
                visual[d-1][j] = b[1][-1 - (d*j)/2]
            else:
                visual[0][j] = b[0][d-1+d*(j/2)]
                for i in range(1,d):
                    visual[i][j] = b[1][-1-i-d*(j/2)]

        file = "block_gibbs_iteration_" + str(itr_num) + ".jpeg"
        self.plot(file,visual)

    def gen_heatmap(self,b,L_heatmap):
        d = self.d
        visual = np.zeros((d,d))
        for j in range(d):
            if j % 2 ==0:
                for i in range(d-1):
                    visual[i][j] = b[0][(d*j)/2 + i]
                visual[d-1][j] = b[1][-1 - (d*j)/2]
            else:
                visual[0][j] = b[0][d-1+d*(j/2)]
                for i in range(1,d):
                    visual[i][j] = b[1][-1-i-d*(j/2)]
        L_heatmap.append(visual.reshape((d*d,)))

    def run(self):
        print "running belief prop..."
        form_samples_shape = (self.d/2, self.d-1)
        p = 0.5

        formatted_samples = np.random.choice(np.array([1,-1]),size=(form_samples_shape),p=[p,1-p])
        L_heatmap = []
        block = 0
        for i in range(self.config['iterations']+1):
            #print "iteration:",i
            #block0
            for j in range(2):
                self.root[j] = self.create_clique_tree(self.d-1)

            self.update_clique_tree(self.root[0],formatted_samples)

            self.upward_msg_pass(self.root[0])

            #self.downward_msg_pass(self.root1)
            formatted_samples,b1_samples = self.get_samples(self.root[0])

            #block1
            self.update_clique_tree(self.root[1],formatted_samples)
            self.upward_msg_pass(self.root[1])
            formatted_samples,b2_samples = self.get_samples(self.root[1])
            self.gen_heatmap([b1_samples,b2_samples],L_heatmap)
            if i%self.config['visualize_checkpt'] == 0:
                print "Running iteration :",i
                self.visualize([b1_samples,b2_samples],i)

        self.plot("block_gibbs_heat_map.jpeg",np.array(L_heatmap))

def run_sub_problem2():
    config = {}
    config['p']                 = 0.5
    config['iterations']        = 1000
    config['visualize_checkpt'] = 100
    config['grid_size']         = 30
    config['theta']             = 0.5
    config['seed']              = 42
    config['name']              = 'vanilla_gibbs'
    gibbs_sampler = gibbs_sampler_class(config)
    gibbs_sampler.run()

def run_sub_problem3():
    # problem config
    config = {}
    config['p']                          = 0.5
    config['iterations']                 = 1000
    config['visualize_checkpt']          = 100
    config['grid_size']                  = 30
    config['theta']                      = 0.5
    config['seed']                       = 42
    config['num_nodes']                  = config['grid_size']
    config['num_messages']               = 4
    config['variable_cardinality']       = 2

    block_gibbs_sampler= block_gibbs_sampler_class(config)
    block_gibbs_sampler.run()


if __name__ == "__main__":
    run_sub_problem2()
    run_sub_problem3()
