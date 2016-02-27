import turtle


class diagramNetwork(object):
    '''Creates a graphic representation of a neural network using turtles.'''
    
   
    def __init__(self,topology,  x_pos = -300, y_pos=200, x_range = 600, y_range=400):
        ''''''
        self.screen = turtle.Screen()
        self.screen.bgcolor('sky blue')
        self.node = turtle.Turtle()
        self.node.hideturtle()
        self.node.up()
        self.node.shape('circle')
        self.node.shapesize(3)
        self.node.color('white')
        self.x = x_pos
        self.y = y_pos
        self.node.setpos(self.x, self.y)
        self.x_range = x_range
        self.y_range = y_range
        self.draw(topology)
        self.screen.exitonclick()
        
        
    def draw(self, topology):
        ''''''
 #       self.node.setpos(self.x, self.y)
        for layer in topology:
#            self.node.stamp()
#            self.x, self.y = self.node.pos()
            for i in range(layer):
                self.node.stamp()
                self.y -= self.y_range/layer
                self.node.setpos(self.x, self.y)
            self.y_range = self.y_range*0.8
            self.y += self.y_range
            self.x += self.x_range/len(topology)
            self.node.setpos(self.x, self.y)
                
        
            
            
            
        
