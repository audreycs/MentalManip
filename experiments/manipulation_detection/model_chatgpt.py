from openai import OpenAI
import logging


class ChatGPTModel:
    def __init__(self,
                 gpt_model,
                 api_key,
                 temperature,
                 top_p,
                 penal,
                 max_input_token_length):
        self.api_key = api_key
        if api_key == "":
            logging.info('Error: OpenAI API key is empty.')
            exit(1)
        self.model_id = gpt_model
        self.client = OpenAI(api_key=self.api_key)
        self.gpt_model = gpt_model
        self.temperature = temperature
        self.top_p = top_p
        self.penal = penal
        self.max_input_token_length = max_input_token_length

    def zeroshot_prompting(self, dialogue):
        system_prompt = """I will provide you with a dialogue. Please determine if it contains elements of mental manipulation. Just answer with 'Yes' or 'No', and don't add anything else.\n"""
        response = self.client.chat.completions.create(
            model=self.gpt_model,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.penal,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": dialogue,
                }
            ]
        )

        res = response.choices[0].message.content

        logging.info(system_prompt)
        logging.info(dialogue)
        logging.info('')
        logging.info(res)
        logging.info('')

        if 'yes' in res.lower():
            return 1
        elif 'no' in res.lower():
            return 0
        else:
            logging.info('Error: response of ChatGPT is neither yes nor no.')
            return -1

    def fewshot_prompting(self, manip_examples, nonmanip_examples, dialogue):
        example_list = []
        total_example_num = len(manip_examples) + len(nonmanip_examples)
        count_example = 0
        for idx, row in manip_examples.iterrows():
            count_example += 1
            example = [
                {"role": "user", "content": f"Example {count_example}:\n{row['Dialogue']}"},
                {"role": "assistant", "content": "Yes"},
            ]
            example_list.extend(example)
        for idx, row in nonmanip_examples.iterrows():
            count_example += 1
            example = [
                {"role": "user", "content": f"Example {count_example}:\n{row['Dialogue']}"},
                {"role": "assistant", "content": "No"},
            ]
            example_list.extend(example)

        system_prompt = f"""I will provide you with a dialogue. Please determine if it contains 
        elements of mental manipulation. Just answer with 'Yes' or 'No', and don't add anything else. 
        Here are {total_example_num} examples:\n"""
        messages = [{"role": "system",
                     "content": system_prompt}]
        messages += example_list
        messages.append({"role": "user",
                         "content": dialogue})

        response = self.client.chat.completions.create(
            model=self.gpt_model,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.penal,
            messages=messages
        )

        res = response.choices[0].message.content

        logging.info(system_prompt)
        for example in example_list:
            logging.info(example['content'])
        logging.info('')
        logging.info(dialogue)
        logging.info('')
        logging.info(res)
        logging.info('')

        if 'yes' in res.lower():
            return 1
        elif 'no' in res.lower():
            return 0
        else:
            logging.info('Error: response of ChatGPT is neither yes nor no.')
            return -1
