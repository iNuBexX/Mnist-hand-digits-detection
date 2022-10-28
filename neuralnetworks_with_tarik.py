import numpy
#import matplotlib.pyplot

import scipy.special

import scipy.misc

# helper to load data from PNG image files
import imageio.v2 as imageio
# glob helps select multiple files using patterns
import glob

import matplotlib.pyplot

# before saying anything i would to deeply thank mr tarik for this wonderfull book and the clear and well elaborated explanation
#this is my first neural network ever following a tutorial from mr tarik using his book called neural networks with tarik 

#img_array = scipy.miscimread("one.png", flatten=True​)

class MyFirstNN:
    def __init__(self,inputnodes, hiddennodes, outputnodes,learningrate):
        self.inodes = inputnodes#input nodes
        self.hnodes = hiddennodes #hidden nodes
        self.onodes = outputnodes#output nodes
        self.lr = learningrate      
        self.wih = numpy.random.normal(0.0, pow(self.hnodes,-0.5),(self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes,-0.5),(self.onodes, self.hnodes))
        self.activation_function = lambda x: scipy.special.expit(x)
        self.inverse_activation_function = lambda x: scipy.special.logit(x)
        pass
    
     
    def train(self,inputs_list,target_list):
        inputs= numpy.array(inputs_list,ndmin=2).T
        targets= numpy.array(target_list,ndmin=2).T
        hidden_inputs=numpy.dot(self.wih,inputs)
        hidden_outputs=self.activation_function(hidden_inputs)
        final_input=numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_input)
        output_errors = targets- final_outputs
        hidden_errors = numpy.dot(self.who.T,output_errors)
        self.who+= self.lr* numpy.dot((output_errors* final_outputs * (1.0- final_outputs)), numpy.transpose(hidden_outputs) )
        self.wih += self.lr * numpy.dot((hidden_errors* hidden_outputs * (1.0-hidden_outputs)),numpy.transpose(inputs))
        
        pass
    def query(self,inputs_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        hidden_input=numpy.dot(self.wih,inputs)
        hidden_output=self.activation_function(hidden_input)
        final_input=numpy.dot(self.who,hidden_output)
        final_outputs = self.activation_function(final_input)
        return final_outputs
       # backquery the neural network
    # we'll use the same termnimology to each item, 
    # eg target are the values at the right of the network, albeit used as input
    # eg hidden_output is the signal to the right of the middle nodes
    def backquery(self, targets_list):
        # transpose the targets list to a vertical array
        final_outputs = numpy.array(targets_list, ndmin=2).T
        
        # calculate the signal into the final output layer
        final_inputs = self.inverse_activation_function(final_outputs)

        # calculate the signal out of the hidden layer
        hidden_outputs = numpy.dot(self.who.T, final_inputs)
        # scale them back to 0.01 to .99
        hidden_outputs -= numpy.min(hidden_outputs)
        hidden_outputs /= numpy.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01
        
        # calculate the signal into the hidden layer
        hidden_inputs = self.inverse_activation_function(hidden_outputs)
        
        # calculate the signal out of the input layer
        inputs = numpy.dot(self.wih.T, hidden_inputs)
        # scale them back to 0.01 to .99
        inputs -= numpy.min(inputs)
        inputs /= numpy.max(inputs)
        inputs *= 0.98
        inputs += 0.01
        
        return inputs 
    

input_nodes = 784 
hidden_nodes = 100 
#i would like to highlight this passage from the book
#"It is worth making an important point here. There isn’t a perfect method for choosing how many hidden nodes there should be for a problem. 
# Indeed there isn’t a perfect method for choosing the number of hidden layers either. 
# The best approaches, for now, are to experiment until you find a good configuration for the problem you’re trying to solve"

output_nodes = 10
# learning rate is 0.3 
learning_rate = 0.3
n = MyFirstNN(input_nodes,hidden_nodes,output_nodes,learning_rate)
##print(n.query([1.0,-0.5,-1.5]))
#now that this basic network is coded lets mnist the hell out of that s*it
training_data_file = open("mnist_dataset/mnist_train.csv", 'r') 
training_data_list = training_data_file.readlines() 
training_data_file.close()





## i would like to highlight this passage from the book 
# They will tell you to read one line at a time and do whatever work you need with each line,and then move onto the next line. 
# They aren’t wrong, it is more efficient to work on a line at a time, and not read the entire file into memory. 
# However our files aren’t that massive, and the code is easier if we use readlines(), and for us, simplicity and clarity is important as we learn Python.


#lets train that baby up
for record in training_data_list: 
    # split the record by the ',' commas 
    all_values = record.split(',') 
    # scale and shift the inputs 
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 
    # create the target output values (all 0.01, except the desired label which is 0.99) 
    targets = numpy.zeros(output_nodes) + 0.01 
    # all_values[0] is the target label for this record    
    targets[int(all_values[0])] = 0.99 
    n.train(inputs, targets) 
    pass

#load the mnist test data CSV file into a list 
test_data_file = open("mnist_dataset/mnist_test.csv", 'r') 
test_data_list = test_data_file.readlines() 
test_data_file.close()




# test the neural network 
# # scorecard for how well the network performs, initially empty 
scorecard = []
#go through all the records in the test data set 
##for record in test_data_list:  #split the record by the ',' commas 
   ## all_values = record.split(',') # correct answer is first value 
  ##  correct_label = int(all_values[0]) 
    #print(correct_label, "correct label") # scale and shift the inputs 
  ##  inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 # query the network 
    ##outputs = n.query(inputs) # the index of the highest value corresponds to the label 
  ##  label = numpy.argmax(outputs) 
    #print(label, "network's answer") # append correct or incorrect to list 
    ##if (label == correct_label): # network's answer matches correct answer, add 1 to scorecard
    ##   scorecard.append(1) 
    ##else:# network's answer doesn't match correct answer, add 0 to scorecard
    ##    scorecard.append(0) 
  ##      pass 
 ##   pass
##
##print(scorecard)
# calculate the performance score, the fraction of correct answers 

# our own image test data set
our_own_dataset = []


for image_file_name in glob.glob('2828_my_own_?.png'):
    
    # use the filename to set the correct label
    label = int(image_file_name[-5:-4])
    
    # load image data from png files into an array
    print ("loading ... ", image_file_name)
    img_array = imageio.imread(image_file_name, as_gray=True)
    
    # reshape from 28x28 to list of 784 values, invert values
    img_data  = 255.0 - img_array.reshape(784)
    
    # then scale data to range from 0.01 to 1.0
    img_data = (img_data / 255.0 * 0.99) + 0.01
    print(numpy.min(img_data))
    print(numpy.max(img_data))
    
    # append label and image data  to test data set
    record = numpy.append(label,img_data)
    our_own_dataset.append(record)
    
    pass
#scorecard_array = numpy.asarray(scorecard) 
#print ("performance = ", scorecard_array.sum() / scorecard_array.size)

# test the neural network with our own images

# record to test
item = 0

# plot image
matplotlib.pyplot.imshow(our_own_dataset[item][1:].reshape(28,28), cmap='Greys', interpolation='None')

# correct answer is first value
correct_label = our_own_dataset[item][0]
# data is remaining values
inputs = our_own_dataset[item][1:]

# query the network
outputs = n.query(inputs)
print (outputs)

# the index of the highest value corresponds to the label
label = numpy.argmax(outputs)
print("network says ", label)
# append correct or incorrect to list
if (label == correct_label):
    print ("match!")
else:
    print ("no match!")
    pass