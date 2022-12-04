import os
import simplejson

import requests


class J1API(object):
    """docstring for GPTJAPI"""
    def __init__(self, large_or_jumbo=False):
        super(J1API, self).__init__()
        self.large_or_jumbo = large_or_jumbo
        self.model_name = 'j1-large' if large_or_jumbo else 'j1-jumbo'
        self.api_key = os.getenv("AI21_API_KEY")

    def query(self, prompt, num_results=16, num_tokens=16):
        res = requests.post(
            "https://api.ai21.com/studio/v1/%s/complete" % self.model_name,
            headers={"Authorization": "Bearer %s" % self.api_key},
            json={
                "prompt": prompt,
                "numResults": num_results,
                "maxTokens": num_tokens,
                "stopSequences": ["."],
                "topKReturn": 64,
                "temperature": 1.0
            }
        )
        try:
            return res.json()
        except simplejson.errors.JSONDecodeError as e:
            print(res)
            print(prompt)
            raise e

    def res_to_completion_text(self, res):
        try:
            return [completion['data']['text'] for completion in res['completions']]
        except KeyError as e:
            print("res:")
            print(res.keys())
            print(res)
            raise e


    def get_log_likelihood(self, prompt, return_sum=False):
        res = self.query(prompt, num_results=1, num_tokens=0)
        tokens = res['prompt']['tokens']
        try:
            if not return_sum:
                return [token['generatedToken'] for token in tokens]
            else:
                return sum([token['generatedToken']['logprob'] for token in tokens])
        except KeyError as e:
            print("res:")
            print(res.keys())
            print(res)
            raise e


if __name__ == "__main__":
    j1 = J1API()
    ll1 = j1.get_log_likelihood(prompt="I love NLP so much", return_sum=True)
    ll2 = j1.get_log_likelihood(prompt="love much so NLP I")
    print("done")