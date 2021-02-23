class N:
	def __init__(self, head, new_head=None, random=None):
		self.head = head
		self.new_head = new_head
		self.random = random

#Print를 위한 재귀 함수
def finding(node):
	if node is None:
		print("null")
		return
	# 현재의 노드 head와 random한 포인터 데이타
	print(node.head, end='')
	if node.random:
		text = "[%d]"%(node.random.head)
		print(text, end = ' -> ')
	else:
		print("[x]", end = ' -> ')

	# 다음 노트를 찾는다
	finding(node.new_head)

def Random_pointer_update(node, dictionary):
	if dictionary.get(node) is None:
		return
	dictionary.get(node).random = dictionary.get(node.random)
	Random_pointer_update(node.new_head, dictionary)

def Recursive(node, dictionary):
	if node is None:
		return None
	dictionary[node] = N(node.head)
	dictionary.get(node).new_head = Recursive(node.new_head, dictionary)
	return dictionary.get(node)

def Clone_linked_list(node):
	dictionaryionary = {}
	Recursive(node, dictionaryionary)
	Random_pointer_update(node, dictionaryionary)
	return dictionaryionary[node]

if __name__ == '__main__':
	node = N(1)
	node.new_head = N(2)
	node.new_head.new_head = N(3)
	node.new_head.new_head.new_head = N(4)
	node.new_head.new_head.new_head.new_head = N(5)

	node.random = node.new_head
	node.new_head.random = node.new_head.new_head
	node.new_head.new_head.random = node.new_head.new_head.new_head
	node.new_head.new_head.new_head.random = node.new_head.new_head.new_head.new_head



	# print("Linked lists:")
	# finding(node)
	clone = Clone_linked_list(node)
	print("\nCloned Linked List:")
	finding(clone)