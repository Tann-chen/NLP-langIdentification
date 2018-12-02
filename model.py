import os
import re
import string
import pickle
import unicodedata

global punctuations
global alphabet
global delta

punctuations = list(string.punctuation)
punctuations += ['¿', '¡', '\n', '\r']  # punctuation for spanish
alphabet = string.ascii_lowercase
delta = 0.5


def read_content(file_path):
	content = None
	with open(file_path, 'r', errors='ignore') as f:
		content = f.read()
	return content


def process_text(content):
	# case fold
	content = content.lower()
	# rm punctuation
	content = re.sub('[' + ''.join(punctuations) + ']', ' ', content)
	# rm digits
	content = re.sub(r'\d+', '', content)
	# tokenize
	tokens = content.split(' ')

	return tokens


def train_bigram_model(all_tokens):
	# init model: nested is after-character
	model = list()
	for i in range(0, 26):
		model.append([delta] * 26)  # smoothing

	# training
	for token in all_tokens:
		token = token.strip()
		
		for idx in range(0, len(token) - 1):
			pre_char = token[idx]
			after_char = token[idx + 1]
			if pre_char not in alphabet or after_char not in alphabet:
				print("[INFO] unvalid alphabet : " + pre_char + after_char)
				continue

			model[alphabet.index(pre_char)][alphabet.index(after_char)] += 1

	# calculate base
	# base = list()
	# for i in range(0, 26):
	# 	all_after_char = model[alphabet[i]]
	# 	count = 0
	# 	for char in all_after_char:
	# 		count += model[alphabet[i]][char]

	# 	base.append(count)
	return model



def train_unigram_model(all_tokens):
	# init model
	model = [delta] * 26

	# training
	for token in all_tokens:
		for idx in range(0, len(token)):
			char = token[idx]
			if char not in alphabet:
				print("[INFO] unvalid alphabet : " + char)
				continue

			model[alphabet.index(char)] += 1

	return model



def remove_diacritics(content):
	return unicodedata.normalize('NFKD', content).encode('ASCII', 'ignore')



if __name__ == '__main__':
	# train english model
	# english_text_1 = read_content('trainset/en-moby-dick.txt')
	# english_text_2 = read_content('trainset/en-the-little-prince.txt')
	# tokens = process_text(english_text_1)
	# tokens += process_text(english_text_2)
	# unigram_model = train_unigram_model(tokens)
	# bigram_model = train_bigram_model(tokens)
	# print(unigram_model)
	# print(bigram_model)


	# train french model
	# french_text_1 = read_content('trainset/fr-le-petit-prince.txt')
	# french_text_2 = read_content('trainset/fr-vingt-mille-lieues-sous-les-mers.txt')
	# tokens = process_text(french_text_1)
	# tokens += process_text(french_text_2)
	# unigram_model = train_unigram_model(tokens)
	# bigram_model = train_bigram_model(tokens)
	# print(unigram_model)
	# print(bigram_model)

	# train spanish model
	spanish_text_1 = remove_diacritics( read_content('trainset/span_germana.txt') )
	print(spanish_text_1)
	#spanish_text_2 = read_content()
	tokens = process_text(spanish_text_1)
	# #tokens += process_text(spanish_text_2)
	# unigram_model = train_unigram_model(tokens) 
	# bigram_model = train_bigram_model(tokens)
	# print(unigram_model)
	# print(bigram_model)







