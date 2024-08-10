#!/usr/bin/env python
# coding: utf-8

# In[32]:


from openai import OpenAI
import os
import sys
client = OpenAI(api_key=os.getenv("AOAI_KEY"))


# In[33]:


# create a function to read text from an ascii file
def read_text(file):
    with open(file, "r") as f:
        return f.read()
    
    


# In[34]:


input= read_text("inputtext.txt")


# In[35]:


import re
input2=re.sub(r'\n+', ' ', input)


# In[36]:


length=len(input2.split())
print(length)


# In[37]:


# print input2 in lines of 90 characters or less
for i in range(0, len(input2), 90):
    print(input2[i:i+90])
    




# In[38]:


response = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "you are a assistant that provides the length of a text and a summary of it in the following JSON format:\n{\n\"Length\": <length of the text in words>,\n\"Summary\": <short summary of the text, maximum 150 words>,\n}\n"
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": input2
        }
      ]
    },
  ],
  temperature=1,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0.53,
  presence_penalty=0,
  response_format={
    "type": "text"
  }
)


# In[39]:


# print the response in a readable format with line breaks
response.choices[0].message.content
#print(response['choices'][0]['message']['content'][0]['text'])


# In[40]:


# split the previous response into sentences
import re
sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', response.choices[0].message.content)
for s in sentences:
    print(s)


# In[ ]:




