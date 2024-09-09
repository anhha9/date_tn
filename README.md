# FLAN-T5-small for normalizing DATE class

As the first part of my dissertation, I extracted DATE examples from the first file of the Sproat & Jaitly's (2017) Google Text Normalization Dataset. 

The format of the resultant dataset has the example following appearance:
input            output
12 Dec           the twelfth of December
20/11            November twentieth 

I fine-tuned FLAN-T5-small model (80M) to explore whether this model can normalize DATE instances only. This prototyping experiment served as the foundation for the next stage in my dissertation, which was to examine if a fine-tuned lightweight LLM can normalize full text. 


During inference, the fine-tuned FLAN-T5-small can normalize DATE class with an accuracy of 97.89%
I expected that when I fine-tuned on the full text (i.e. with more context), the accuracy could increase. 
