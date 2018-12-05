import os
import re
import math
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

	# calculate possib
	for pre in range(0, 26):
		all_after_char = model[pre]
		count = sum(all_after_char)

		for after in range(0, 26):
			model[pre][after] = model[pre][after] / count

	return model



def train_unigram_model(all_tokens):
	# init model
	model = [delta] * 26
	total_count = delta * 26

	# training
	for token in all_tokens:
		for idx in range(0, len(token)):
			char = token[idx]
			if char not in alphabet:
				print("[INFO] unvalid alphabet : " + char)
				continue

			model[alphabet.index(char)] += 1
			total_count += 1

	# calculate possib
	for i in range(0, 26):
		model[i] = model[i] / total_count

	return model



def remove_diacritics(content):
	return unicodedata.normalize('NFKD', content).encode('ASCII', 'ignore').decode('utf-8')



def write_unigram_2_file(unigram_model, file_path):
	content = ''
	for i in range(0, 26):
		alpha = alphabet[i]
		content += 'P(' + alpha + ') = ' + str(unigram_model[i]) + '\n'

	with open(file_path, 'w') as f:
		f.write(content)



def write_bigram_2_file(bigram_model, file_path):
	content = ''
	for pre in range(0, 26):
		for after in range(0, 26):
			content += 'P(' + alphabet[after] + '|' + alphabet[pre] + ') = ' + str(bigram_model[pre][after]) + '\n'

	with open(file_path, 'w') as f:
		f.write(content)



def pred_by_unigram(en_unigram_model, fr_unigram_model, span_unigram_model, sentence):
	en_unigram_possib = 0
	fr_unigram_possib = 0
	span_unigram_possib = 0
	tokens = process_text(sentence)

	for t in tokens:
		t = t.strip()
		for i in range(0, len(t)):
			alpha = t[i]
			alpha_index = alphabet.index(alpha)
			
			fr_unigram_possib += math.log10(fr_unigram_model[alpha_index])
			en_unigram_possib += math.log10(en_unigram_model[alpha_index])
			span_unigram_possib += math.log10(span_unigram_model[alpha_index])

	# get max
	if en_unigram_possib >= fr_unigram_possib and en_unigram_possib >= span_unigram_possib:
		return 'en'
	elif fr_unigram_possib >= en_unigram_possib and fr_unigram_possib >= span_unigram_possib:
		return 'fr'
	elif span_unigram_possib >= fr_unigram_possib and span_unigram_possib >= en_unigram_possib:
		return'span'



def pred_by_bigram(en_bigram_model, fr_bigram_model, span_bigram_model, sentence):
	en_bigram_possib = 0
	fr_bigram_possib = 0
	span_bigram_possib = 0
	tokens = process_text(sentence)

	for t in tokens:
		t = t.strip()

		for i in range(0, len(t) - 1):
			pre_char = t[i]
			after_char = t[i + 1]
			pre_index = alphabet.index(pre_char)
			after_index = alphabet.index(after_char)
							
			fr_bigram_possib += math.log10(fr_bigram_model[pre_index][after_index])
			en_bigram_possib += math.log10(en_bigram_model[pre_index][after_index])
			span_bigram_possib += math.log10(span_bigram_model[pre_index][after_index])
		

	# get max
	if en_bigram_possib >= fr_bigram_possib and en_bigram_possib >= span_bigram_possib:
		return 'en'
	elif fr_bigram_possib >= en_bigram_possib and fr_bigram_possib >= span_bigram_possib:
		return 'fr'
	elif span_bigram_possib >= fr_bigram_possib and span_bigram_possib >= en_bigram_possib:
		return 'span'




if __name__ == '__main__':
	# train english model
	english_text_1 = read_content('trainset/en-moby-dick.txt')
	english_text_2 = read_content('trainset/en-the-little-prince.txt')
	tokens = process_text(english_text_1)
	tokens += process_text(english_text_2)
	en_unigram_model = train_unigram_model(tokens)
	write_unigram_2_file(en_unigram_model, 'unigramEN.txt')
	en_bigram_model = train_bigram_model(tokens)
	write_bigram_2_file(en_bigram_model, 'bigramEN.txt')

	del tokens

	# train french model
	french_text_1 = read_content('trainset/fr-le-petit-prince.txt')
	french_text_2 = read_content('trainset/fr-vingt-mille-lieues-sous-les-mers.txt')
	tokens = process_text(french_text_1)
	tokens += process_text(french_text_2)
	fr_unigram_model = train_unigram_model(tokens)
	write_unigram_2_file(fr_unigram_model, 'unigramFR.txt')
	fr_bigram_model = train_bigram_model(tokens)
	write_bigram_2_file(fr_bigram_model, 'bigramFR.txt')

	del tokens


	# train spanish model
	spanish_text_1 = remove_diacritics(read_content('trainset/span-germana.txt'))
	spanish_text_2 = remove_diacritics(read_content('trainset/span-La-nariz-de-un-notario.txt')) 
	tokens = process_text(spanish_text_1)
	tokens += process_text(spanish_text_2)
	span_unigram_model = train_unigram_model(tokens)
	write_unigram_2_file(span_unigram_model, 'unigramOT.txt')
	span_bigram_model = train_bigram_model(tokens)
	write_bigram_2_file(span_bigram_model, 'bigramOT.txt')


	IF_PREDICT = True
	IF_EXPERIMENT = False

	# ------------------ predict ------------------
	if IF_PREDICT:
		pred_content = read_content('input.txt')
		sentences = pred_content.split('\n')
		sentence_index = 1
		 
		for s in sentences:
			en_unigram_possib = 0
			fr_unigram_possib = 0
			span_unigram_possib = 0
			
			log = s + '\r\r'

			# preprocess sentence
			tokens = process_text(s)

			log += 'UNIGRAM MODEL:\r\r'
			
			for t in tokens:
				t = t.strip()
				for i in range(0, len(t)):
					alpha = t[i]
					alpha_index = alphabet.index(alpha)
					
					log += 'UNIGRAM: ' + alpha + '\r'
					fr_unigram_possib += math.log10(fr_unigram_model[alpha_index])
					log += 'FRENCH: P(' + alpha + ') = ' + str(fr_unigram_model[alpha_index]) + ' ==> log prob of sentence so far: ' + str(fr_unigram_possib) + '\r'
					en_unigram_possib += math.log10(en_unigram_model[alpha_index])
					log += 'ENGLISH: P(' + alpha + ') = ' + str(en_unigram_model[alpha_index]) + ' ==> log prob of sentence so far: ' + str(en_unigram_possib) + '\r'
					span_unigram_possib += math.log10(span_unigram_model[alpha_index])
					log += 'OTHER: P(' + alpha + ') = ' + str(span_unigram_model[alpha_index]) + ' ==> log prob of sentence so far: ' + str(span_unigram_possib) + '\r\r'

			# get max
			if en_unigram_possib >= fr_unigram_possib and en_unigram_possib >= span_unigram_possib:
				log += 'According to the unigram model, the sentence is in English'
			elif fr_unigram_possib >= en_unigram_possib and fr_unigram_possib >= span_unigram_possib:
				log += 'According to the unigram model, the sentence is in French'
			elif span_unigram_possib >= fr_unigram_possib and span_unigram_possib >= en_unigram_possib:
				log += 'According to the unigram model, the sentence is in Spanish'


			log += '\r----------------\r'
			log += 'BIGRAM MODEL:\r\r'

			en_bigram_possib = 0
			fr_bigram_possib = 0
			span_bigram_possib = 0

			for t in tokens:
				t = t.strip()

				for i in range(0, len(t) - 1):
					pre_char = t[i]
					after_char = t[i + 1]
					pre_index = alphabet.index(pre_char)
					after_index = alphabet.index(after_char)
					
					log += 'BIGRAM: ' + pre_char + after_char + '\r'				
					fr_bigram_possib += math.log10(fr_bigram_model[pre_index][after_index])
					log += 'FRENCH: P(' + after_char + '|' + pre_char + ') = ' + str(fr_bigram_model[pre_index][after_index]) + ' ==> log prob of sentence so far: ' + str(fr_bigram_possib) + '\r'
					en_bigram_possib += math.log10(en_bigram_model[pre_index][after_index])
					log += 'ENGLISH: P(' + after_char + '|' + pre_char + ') = ' + str(en_bigram_model[pre_index][after_index]) + ' ==> log prob of sentence so far: ' + str(en_bigram_possib) + '\r'
					span_bigram_possib += math.log10(span_bigram_model[pre_index][after_index])
					log += 'OTHER: P(' + after_char + '|' + pre_char + ') = ' + str(span_bigram_model[pre_index][after_index]) + ' ==> log prob of sentence so far: ' + str(span_bigram_possib) + '\r\r'

			# get max
			if en_bigram_possib >= fr_bigram_possib and en_bigram_possib >= span_bigram_possib:
				log += 'According to the bigram model, the sentence is in English'
			elif fr_bigram_possib >= en_bigram_possib and fr_bigram_possib >= span_bigram_possib:
				log += 'According to the bigram model, the sentence is in French'
			elif span_bigram_possib >= fr_bigram_possib and span_bigram_possib >= en_bigram_possib:
				log += 'According to the bigram model, the sentence is in Spanish'

			# write log
			with open('out' + str(sentence_index) + '.txt', 'w') as f:
				f.write(log)
				sentence_index += 1



# -------------- experiment --------------
	if IF_EXPERIMENT:
		test_data = french_text_1 + french_text_2
		sentences = re.split('[!.?]', test_data)
		true = 'fr'

		unigram_error_sentences = list()
		bigram_error_sentences = list()
		both_correct_sentences = list()
		both_error_sentences = list()

		for s in sentences:
			unigram_pred = pred_by_unigram(en_unigram_model, fr_unigram_model, span_unigram_model, s)
			bigram_pred = pred_by_bigram(en_bigram_model, fr_bigram_model, span_bigram_model, s)

			if unigram_pred != true:
				unigram_error_sentences.append(s)
			
			if bigram_pred != true:
				bigram_error_sentences.append(s)

			if unigram_pred != true and bigram_pred != true:
				both_error_sentences.append(s)

			if unigram_pred == true and bigram_pred == true:
				both_correct_sentences.append(s)

		print('-------- end --------')
		print('total : ' + str(len(sentences)))
		print('unigram accuracy : ' + str(1 - (len(unigram_error_sentences) / len(sentences))))
		print('bigram accuracy : ' + str(1 - (len(bigram_error_sentences) / len(sentences))))
		# get top 5 sentences of both error sentences
		# print("----- error pred ----")
		# print(both_error_sentences)
		# print("------- correct pred ------")
		# print(both_correct_sentences[: 20])
















