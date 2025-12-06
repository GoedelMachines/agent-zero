class SessionManager:
	"""
	Class to handle session management. Create this the first layer which infers from the context to dump the user into a session
	It needs to use the contextual memory from all other sessions as well.

	Args:
		`current_input`: The recent input made by the user
	"""

	def __init__(self, current_input: str):
