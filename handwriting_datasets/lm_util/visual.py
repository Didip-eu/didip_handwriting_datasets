
class Viz():

    def __init__(self):
        pass


    def show(self, states, N, L, C ):
        """
        Visualize hidden states 

        """
        # look at a given batch (say, batch 0), given height
        print(states[0,L/4,:])
        print(states[0,/2,:])
        print(states[0,3*L/4,:])




