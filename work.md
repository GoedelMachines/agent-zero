- [x] Make an abstraction for maya1. Just need to give out an API for `synthesize_sentences` and make sure that is available to the above models.

- [ ] Undestand how the memory part works and how each chat's separate memory is saved so that I can nicely place my function for retrieving the memory and checking which one it belongs to.

```
	A higher level LLM to differentiate between chats, to start a new instance or to continue this instance. 
	This needs to be active all the time. (in the background). This has the context of “what’s happening at a higher level” in each chat. And we should be able to start a new instance mid way (so this is always listening).
	from `python/helpers/memory.py`, it contains functions for intialising a vectorDB which is managed using `faiss`. This is a vectorDB management tool released by facebook. Nice.
```

We'll take this step by step:

- [ ] Find a way to pass the prompt to the main agent first, put the main guy in the loop. Let's start with that.