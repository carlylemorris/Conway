import numpy as np
import time

#https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life


class LifeGame:

    #helper funcs
    def mask(self,r,c):#ret flattened life mask of neighbors size for board[r][c]

        out = np.zeros((self.HIGH,self.WIDE))   #TODO look into making sparse

        for i in range(r-1,r+2,1):
            for j in range(c-1,c+2,1):
                if 0 <= i < self.HIGH and 0 <= j < self.WIDE:
                    out[i][j] = 1
        
        out[r][c] = 0

        return out.flatten()

    liveRules = np.vectorize(lambda n : int(2 <= n <= 3))

    deadRules = np.vectorize(lambda n : int(n == 3))

    negate = np.vectorize(lambda n : int(not bool(n))) #numpy didnt like n^1

    populate = np.vectorize(lambda n,threshold : int(n>=threshold))#from old seed implementation

    #initilize shape-dependent objects
    def __init__(self,heigth,width):
        self.HIGH = heigth
        self.WIDE = width

        #neighbor mask 
        neighbors = list()
        for i in range(self.HIGH):
            for j in range(self.WIDE):
                neighbors.append(self.mask(i,j))

        self.neighbors = np.array(neighbors).reshape((self.HIGH*self.WIDE,self.HIGH*self.WIDE))
        
        #border mask
        endzone = list()

        endzone.append(np.ones(self.WIDE))
        for i in range(self.HIGH-2):
            temp = np.zeros(self.WIDE)
            temp[0] = 1
            temp[-1] = 1
            endzone.append(temp)
        endzone.append(np.ones(self.WIDE))

        self.endzone = np.array(endzone).reshape((self.HIGH,self.WIDE))


    #update game state by 1 timestep
    #O(n^2) because of dot product
    def step(self,board):
        vBoard = board.flatten()

        #compute pop density per square
        pop = np.dot(self.neighbors,vBoard)#vBoard broadcasts to (high,wide,size(dots))
        assert(pop.size == self.HIGH*self.WIDE)

        livePop = np.multiply(pop,vBoard)#separate so two different criteria can be applied

        deadPop = np.multiply(pop,self.negate(vBoard))

        vBoard = np.add(self.liveRules(livePop),self.deadRules(deadPop))

        return vBoard.reshape((self.HIGH,self.WIDE))


    def isSoln(self,board,steps=0,onStep=lambda x:x):#does a state reach the edge in n steps
        assert(board.shape==(self.HIGH,self.WIDE))

        if not steps:#sensible default, cant acess self namespace in function signature
            steps = max(self.HIGH,self.WIDE) * 2

        past = set()

        soln = False
        for i in range(steps):
            if(np.sum(np.multiply(board,self.endzone)) > 0):
                soln = True
                break
            
            if(board.data.tobytes() in past):#life is state func so any loops are infinite. This also catches 0 board -> 0 board case
                break
                
            past.add(board.data.tobytes())

            board = self.step(board)

            onStep(board)
        
        return soln #assume false after maxsteps
        
    #get a random puzzle (might not be solvable)
    def genBoard(self,seedSize=4): 
        
        board = np.zeros((self.HIGH,self.WIDE),dtype="int")

        seed = np.random.uniform(size=(seedSize,seedSize))

        yMid = self.HIGH//2
        xMid = self.WIDE//2

        s2b = lambda p :    (yMid - seedSize//2 + (p // seedSize),     #maps from seed coord (scalar from argmin()) to desired board coords
                             xMid - seedSize//2 + (p % seedSize))       

        i = 0
        solns = list()
        while(i < seedSize**2 and len(solns) == 0):
            p = np.argmin(seed)
            board[s2b(p)] = 1
            seed[p//seedSize][p%seedSize] = 1
            i += 1

            #find if there are solutions
            canChange = np.dot(self.neighbors,board.flatten())
            np.dot(self.neighbors,canChange,out=canChange)
            canChange.shape = (self.HIGH,self.WIDE)

            for r in range(yMid - seedSize//2 -2,     yMid - seedSize//2 + seedSize +2):#range of tiles that can affect seed
                for c in range(xMid - seedSize//2 - 2,     xMid - seedSize//2 + seedSize +2):   
                        if canChange[r][c]:         #Only run sumulation if the changed cell can interact with current cells
                            board[r][c] = int(not bool(board[r][c]))
                            if self.isSoln(board):
                                solns.append((r,c))
                            board[r][c] = int(not bool(board[r][c]))

        return(board,solns)

    def prettyPrint(self,board):
        out = " "*3
        for c in range(board.shape[1]):
            out += f"{c:2d}|"

        for r in range(board.shape[0]):
            out+="\n"+f"{r:2d}|"
            for c in range(board.shape[1]):
                if board[r][c]:
                    out += " # "
                else:
                    out += " - "
        print(out+"\n")

    def play(self):
        solved = False
        board, solns = self.genBoard(min(self.HIGH,self.WIDE)//2-1)

        while len(solns) < 1:
            board, solns = self.genBoard(min(self.HIGH,self.WIDE)//2-1)

        while True:
            print("Flip one cell to reach the edge")
            self.prettyPrint(board)

            row = 0
            col = 0
            while not (0 < row < self.HIGH-1 and 0 < col < self.WIDE-1):
                try:
                    raw = input("> ")
                    row = int(raw.split(" ")[0])
                    col = int(raw.split(" ")[1])
                except:
                    if raw == "q":
                        return solved
                    row=0
                    col=0
                    print(f"Enter two coordinates between (1,1) and ({self.HIGH-1},{self.WIDE-1}) separated with a space")
                    print("Like:\n> 12 6")

            def printWait(board):
                self.prettyPrint(board)
                time.sleep(1)
                

            board[row][col] = int(not bool(board[row][col]))

            self.prettyPrint(board)
            if self.isSoln(board,onStep=printWait):

                print("\n\n\nYou won!\n\n\n")
                solved = True
                solns.remove((row,col))

                if len(solns) > 0:
                    print(f"There are {len(solns)} more solutions, keep playing? y/n")
                    if input() != "y":
                        return solved
                else:
                    return solved

            else:
                print("Not quite!")
            

            board[row][col] = int(not bool(board[row][col]))
            


            



        
    


def main():
    size = 10
    while True:
        life=LifeGame(size,size)
        if life.play():
            size = size+1
        else:
            quit()


if __name__ == "__main__":
    main()

