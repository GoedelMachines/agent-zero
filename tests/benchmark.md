# Benchmarking maya1 for time taken

Default settings:

            max_new_tokens=4096,  # Increase to let model finish naturally
            min_new_tokens=28,  # At least 4 SNAC frames
            temperature=0.1, 
            top_p=0.9, 
            repetition_penalty=1.1,  # Prevent loops
            do_sample=True,
            eos_token_id=CODE_END_TOKEN_ID,  # Stop at end of speech token
            pad_token_id=tokenizer.pad_token_id,

Runs:

	test1: # we don't care about the CPU usage
	input_tokens = 33
	generated_tokens = 246
	time_taken = 55.8
	
	test2:
	input_tokens = 33
	generated_tokens = 267
	time_taken = 48.6
	
	test3:
	input_tokens = 33
	generated_tokens = 239
	time_taken = 46.1

	average for 33 input tokens:
	generated_tokens = 250
	time_taken = 51

	Cooked only. No point doing more because this is too slow as it is. It might get faster with a GPU however.	