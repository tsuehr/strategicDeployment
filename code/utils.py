import numpy as np
import matplotlib.pyplot as plt
import math


def ccf_old(ml_utility, max_utility):
    if ml_utility<=0:
        return None
    else:
        return min(max_utility, ml_utility*1.5)
    
def get_discounted_utility(discount_factor,utilityseries):
    utility = 0
    # print(discount_factor)
    # print(utilityseries)
    for i in range(0,len(utilityseries)):
        utility = utility + utilityseries[i] * (1+discount_factor)**(-i)
    return utility

def ccf(ml_utility,max_utility, ccf_beta=3.0):
    if ml_utility<0:
        return 0.0
    elif ml_utility<=0.25:
        #return 0.25+0.1*math.sin(500*math.pi*25*ml_utility)
        return 0.4
    else:
        return min(max_utility, 1/(np.exp(-ccf_beta*ml_utility)+1))

def get_error(data_size, c=1.0):
    if data_size<=0:
        return None
    else:
        return c/math.sqrt(data_size)

def get_angle(error, EPS=0.0000005):
    angle = 0
    if error >1:
        return 0
    if error <0:
        return 90
    
    if error <= EPS: #perfect learning
        angle = 90
    elif error >=1: # usually we can assume that 1/sqrt(n) <= 1 
        angle = 0
    else:
        angle = (1-error)*90
        
    return angle

def calculate_intersection(alpha, current_utility):
    alpha_rad = np.radians(alpha)

    #slope of the learning trajectory
    if alpha == 90:
        m = 0 
    else:
        m = -1 / np.tan(alpha_rad)

    #y = m * x + current_utility
    #diagonal: y = x
    #set equal: m * x + current_utility = x
    #=> (m - 1) * x = -current_utility
    if m != 1:
        x_intersection = -current_utility / (m - 1)
        y_intersection = x_intersection  #y = x
        return x_intersection, y_intersection
    else:
        return None  #no intersection
    

class ContinuousEnvironment:
    def __init__(self, params):
        self.current_utility = params["start_utility"]
        self.next_utility = self.current_utility
        self.time_preference = params["time_preference"]
        self.current_datasize = 0
        self.data_collection_size = params["data_collection_size"]
        self.accumulated_utility = []
        self.current_error = 1
        self.max_utility = params["max_utility"]
        self.points = []
        self.next_ml_utility = 0
        self.next_intersections = 0
        self.intersections = []
        self.next_angle = 0
        self.angles = []
        self.ccf_beta = params["ccf_beta"]
        self.filename = params["filename"]
        self.data_variance = params["data_variance"]
        
    def collect_data(self):
        self.current_datasize = self.current_datasize+self.data_collection_size
        return self.current_datasize

    def step(self, action):
        if action == "collect":
            self.accumulated_utility.append(self.current_utility)
            self.current_datasize = self.collect_data()
            self.current_error = get_error(self.current_datasize, c=self.data_variance)
            learning_trajectory = get_angle(self.current_error)
            self.next_angle = learning_trajectory
            self.next_ml_utility,_ = calculate_intersection(learning_trajectory,self.current_utility)
            self.next_utility = ccf(self.next_ml_utility,self.max_utility, ccf_beta=self.ccf_beta)
            self.intersection = self.next_ml_utility
        elif action == "deploy":
            #print(self.current_error)
            self.accumulated_utility.append(self.current_utility)
            self.current_datasize = 0
            self.current_utility = self.next_utility
            self.current_error = 1.0
            self.points.append((self.next_ml_utility, self.current_utility))
            self.intersections.append(self.next_ml_utility)
            self.angles.append(self.next_angle)
            self.next_angle = 0
        else:
            print(f"Action ´´{action}´´ not found")
        #print(f"Current Utility {self.current_utility}")
        #print(f"Next Utility {self.next_utility}")

    def render(self, other_filename=None):
        #diagonal f(x) = x
        fig, ax = plt.subplots(figsize=(10,7))

        x_vals = np.linspace(0, 1, 100)
        y_vals = x_vals
        
        x,y = zip(*self.points)
        ccf_vals = [ccf(xs,1.0, self.ccf_beta) for xs in x_vals]
        ax.plot(x_vals, y_vals, label="Diagonal f(x) = x", linestyle="--")
        ax.scatter(x,y, color='red')
        ax.scatter(self.intersections, self.intersections, color='green')
        #ax.plot(x,y, color='red', linestyle="--")
        ax.plot(x_vals,ccf_vals, color='red', linestyle="--")
        ys = [item for pair in zip(self.intersections,y) for item in pair]
        xs = [item for item in x for _ in range(2)]
        ax.plot(xs,ys, color='blue')
        
        plt.ylabel("ML+Human Utility")
        plt.xlabel("ML Utility")
        plt.title("Deployment Strategy")
        plt.grid()
        #plt.show()
        if other_filename:
            plt.savefig(other_filename, format='png', dpi=300)
        plt.savefig(self.filename, format='png', dpi=300)
        plt.close(fig)
