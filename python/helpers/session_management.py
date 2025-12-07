class SessionManager:
	"""
	Class to handle session management. Create this the first layer which infers from the context to dump the user into a session
	It needs to use the contextual memory from all other sessions as well.

	Args:
		`current_input`: The recent input made by the user

	How does it handle the sessions?

	- uses the CreateChat class from create_chat.py. I THINK we simply need to call .process() from it.
	- It needs better OVERALL memory indexed by the chat instance. It checks which memory bracket the current prompt fits into
	then pushes it then. 
	- It also has specific "defintely turn on" commands. As in, if I explicitly tell it to create a new context, then it needs to do that, without fail
	
	I need to first make memory, or understand how its doing memory
	"""

	def __init__(self, current_input: str):
