#!/usr/bin/env python
# coding: utf-8

# In[2]:


from openai import OpenAI
import os
import sys
client = OpenAI(api_key=os.getenv("AOAI_KEY"))


# In[36]:


# create a function to read text from an ascii file
def read_text(file):
    with open(file, "r") as f:
        return f.read()
    
    


# In[37]:


input= read_text("inputtext.txt")


# In[38]:


import re
input2=re.sub(r'\n+', ' ', input)


# In[39]:


# print input2 in lines of 90 characters or less
for i in range(0, len(input2), 90):
    print(input2[i:i+90])
    




# In[40]:


response = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "you are a assistant that provides the date of the review and a summary of text in the following JSON format:\n{\n\"Date\": <data of the review>,\n\"Summary\": <short summary of the text, maximum 50 words>,\n}\n"
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
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "{\n\"Date\": \"N/A\",\n\"Summary\": \"Primary waves experience one reflection, while multiples have several. Multiples are categorized into surface and internal types. Surface multiples benefit marine data; internal multiples are valuable for processing land data.\"\n}"
        }
      ]
    }
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


# In[41]:


# print the response in a readable format with line breaks
response.choices[0].message.content
#print(response['choices'][0]['message']['content'][0]['text'])


# In[42]:


# split the previous response into sentences
import re
sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', response.choices[0].message.content)
for s in sentences:
    print(s)


# In[ ]:




