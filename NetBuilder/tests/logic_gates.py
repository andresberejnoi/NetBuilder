from ..NeuralNet import Network


def test_AND():
    #training set for logic AND 
    aand =  [[np.array([-0.5,-0.5]), np.array([-0.5])],
             [np.array([-0.5,0.5]), np.array([-0.5])],
             [np.array([0.5,-0.5]), np.array([-0.5])],
             [np.array([0.5,0.5]), np.array([0.5])]]

    aand = [(array[0],array[1]) for array in aand]
    
    
    #Setting up the network
    topology = [2,10,10,1]
    epochs = 1000
    tolerance = 1E-10
    trainingSet = annd
    
    print("="*80)
    print("Training...\n")
    net = Network(topology, learningRate=0.1, momentum=0.1)
    net.train(trainingSet,epochs,tolerance, batch=False)         #training begins
    
    #Now, show the results of training
    #It would be better to create a function to display this information in a better way
    #print("="*80)      #will 80 '=' signs to separate the line
    print()
    print("Testing network:")
    print("INPUTS    |\tPREDICTION\t   | EXPECTED")
    for inputs,target in trainingSet:
        out = net.feedforward(inputs)
    
        print("{0} {1} \t {2} \t\t\t {3}   ".format(inputs[0],inputs[1],out[0],target[0]))              #for some reason, the last line is not tabbed in
        
    print("="*80)
    
    #Saving the Network 
    #extracting the value of the training pattern:
    #and_patterns = [pat[0] for pat in annd]
    #save_outputs("AND_outs.csv", and_patterns, net)
    
def test_OR():
    #training set for logic OR 
    oor = [ [np.array([-0.5,-0.5]), np.array([-0.5])],
            [np.array([-0.5,0.5]), np.array([0.5])],
            [np.array([0.5,-0.5]), np.array([0.5])],
            [np.array([0.5,0.5]), np.array([0.5])]]

    oor = [(array[0],array[1]) for array in oor]
    
    
    #Setting up the network
    topology = [2,10,10,1]
    epochs = 1000
    tolerance = 1E-10
    trainingSet = oor
    
    print("="*80)
    print("Training...\n")
    net = Network(topology, learningRate=0.1, momentum=0.1)
    net.train(trainingSet,epochs,tolerance, batch=False)         #training begins
    
    #Now, show the results of training
    #It would be better to create a function to display this information in a better way
    #print("="*80)      #will 80 '=' signs to separate the line
    print()
    print("Testing network:")
    print("INPUTS    |\tPREDICTION\t   | EXPECTED")
    for inputs,target in trainingSet:
        out = net.feedforward(inputs)
    
        print("{0} {1} \t {2} \t\t\t {3}".format(inputs[0],inputs[1],out[0],target[0]))              #for some reason, the last line is not tabbed in
        
    print("="*80)
    
    #Saving the Network 
    #extracting the value of the training pattern:
    #or_patterns = [pat[0] for pat in oor]
    #save_outputs("OR_outs.csv", or_patterns, net)
    
def test_XOR():
    #training set for logic OR 
    xor = [
            [np.array([-0.5,-0.5]), np.array([-0.5])],
            [np.array([-0.5,0.5]), np.array([0.5])],
            [np.array([0.5,-0.5]), np.array([0.5])],
            [np.array([0.5,0.5]), np.array([-0.5])]
          ]

    xor = [(array[0],array[1]) for array in xor]
    
    
    #Setting up the network
    topology = [2,10,10,1]
    epochs = 1000
    tolerance = 1E-10
    trainingSet = oor
    
    print("="*80)
    print("Training...\n")
    net = Network(topology, learningRate=0.1, momentum=0.1)
    net.train(trainingSet,epochs,tolerance, batch=False)         #training begins
    
    #Now, show the results of training
    #It would be better to create a function to display this information in a better way
    #print("="*80)      #will 80 '=' signs to separate the line
    print()
    print("Testing network:")
    print("INPUTS    |\tPREDICTION\t   | EXPECTED")
    for inputs,target in trainingSet:
        out = net.feedforward(inputs)
    
        print("{0} {1} \t {2} \t\t\t {3}".format(inputs[0],inputs[1],out[0],target[0]))              #for some reason, the last line is not tabbed in
        
    print("="*80)
    
    #Saving the Network 
    #extracting the value of the training pattern:
    #xor_patterns = [pat[0] for pat in xor]
    #save_outputs("XOR_outs.csv", xor_patterns, net)
