input = [[10,3,6,4],[1,5,8,0],[2,13,7,15],[14,9,12,11]]

coordinates = [(1,1), (1,2), (1,3), (1,4),
               (2,1), (2,2), (2,3), (2,4),
               (3,1), (3,2), (3,3), (3,4),
               (4,1), (4,2), (4,3), (4,4)]
endState = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0]
startState = [1,2,3,4,5,0,6,8,9,10,7,11,13,14,15,12]
endCoordinates = dict(zip(coordinates, endState))
currentCoordinates = dict(zip(coordinates, startState))
spaceKey = None
possiblePaths = {}
closedPaths = []
finalPath = []
# TODO: adjust code to make it viable for inputs of different sizes. Figure out why algorithm isn't working out complicated paths. Determine solvability using inversions
# def generateEnd():
# 	return dict(zip(coordinates, endState))
#
# def generateStart():
# 	return dict(zip(coordinates, startState))

class InvalidMoveException(Exception):
	def __init__(self, message, exception):
		super().__init__(message + exception)

		self.exception = exception



#Moving functions
def moveSpace(cmd, space, dictionary): # Moves the space up, right, down, or left
	action = None
	y, x = space

	if cmd == 'moveUp': action = (y - 1, x)
	if cmd == 'moveRight': action = (y, x + 1)
	if cmd == 'moveDown': action = (y + 1, x)
	if cmd == 'moveLeft': action = (y, x - 1)

	assert action is not None

	if action in dictionary:
		temp =  dictionary[action]
		pemt = dictionary[(y, x)]
		dictionary[(y, x)] = temp
		dictionary[action] = pemt
		return action    #make sure this is updated properly
	else:
		raise InvalidMoveException("invalid move:", cmd )

def dictToArray(dictionary):

	arr = []
	for y in range(4):
		tmp = []
		for x in range(4):
			tmp += [dictionary[(y+1, x+1)]]
		arr += [tmp]

	# for row in arr:
	# 	print(row)

	return arr


def possibleTransitions(space, dictionary):
	y, x = space
	# closedPaths.append(spaceKey[0]) #will need to print the actual value
	tempPath = currentCoordinates

	if (y - 1, x) in dictionary:
		possiblePaths['moveUp'] = 0
	if (y, x + 1) in dictionary:
		possiblePaths['moveRight'] = 0
	if (y + 1, x) in dictionary:
		possiblePaths['moveDown'] = 0
	if (y, x - 1) in dictionary:
		possiblePaths['moveLeft'] = 0

def flushTransitions():
	possiblePaths.clear()

def calculateHScore(start, finish):
	current = dictToArray(start)
	goal =  dictToArray(finish)

	count = 0
	for a, b in zip(current, goal): # iterate through both elements at the same time.
		for c, d in zip(a,b):
			if c == d:
				continue
			else:
				count += 1
	return count



if __name__ == "__main__":

	spaceKey = [key for key, value in currentCoordinates.items() if value == 0][0]
	print(dictToArray(currentCoordinates))
	g = 0
	while currentCoordinates != endCoordinates:
		flushTransitions()
		possibleTransitions(spaceKey, currentCoordinates)
		g += 1
		for k in possiblePaths:
			duplicate = dict(currentCoordinates)
			moveSpace(k, spaceKey, duplicate)
			possiblePaths[k] = (str(dictToArray(duplicate)), g + calculateHScore(duplicate, endCoordinates))

		# closedPaths.append()
		nextMove = min(possiblePaths.values(), key = possiblePaths.get)
		newSpace = moveSpace(nextMove, spaceKey, currentCoordinates)
		finalPath.append(currentCoordinates.get(spaceKey))
		spaceKey = newSpace
		print(dictToArray(currentCoordinates))

	print(finalPath)
