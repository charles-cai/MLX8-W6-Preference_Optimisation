Some progress update:

_ I trained SFT with 20K train items (should be from original Dataloader.py) and as a LoRA adapter. I think there is some success as some summaries are quite promising and complete.
- I spent time working on SFT'd model, inferring it to get expected behavior out of it.
- I trained the reward model with combination to dataloaderV2 or V3. Also on 20K items from dataset,
- validation.py is just a pipeline to load test data and loop a few test summary examples...
- Didn't test the reward model yet but hoping to get a Policy Model trained overnight to at least have covered the full pipeline regardless of results.
- PPO to be continued...
