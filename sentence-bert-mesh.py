from transformers import BertJapaneseTokenizer, BertModel
import torch
import scipy.spatial
import numpy as np


# 以下のqiita記事を参考に
# https://qiita.com/sonoisa/items/1df94d0a98cd4f209051
class SentenceBertJapanese:
	def __init__(self, model_name_or_path, device=None):
		self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
		self.model = BertModel.from_pretrained(model_name_or_path)
		self.model.eval()

		if device is None:
			device = "cuda" if torch.cuda.is_available() else "cpu"
		self.device = torch.device(device)
		self.model.to(device)

	def _mean_pooling(self, model_output, attention_mask):
		token_embeddings = model_output[0] #First element of model_output contains all token embeddings
		input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
		return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

	@torch.no_grad()
	def encode(self, sentences, batch_size=8):
		all_embeddings = []
		iterator = range(0, len(sentences), batch_size)
		for batch_idx in iterator:
			batch = sentences[batch_idx:batch_idx + batch_size]

			encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
										   truncation=True, return_tensors="pt").to(self.device)
			model_output = self.model(**encoded_input)
			sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

			all_embeddings.extend(sentence_embeddings)

		# return torch.stack(all_embeddings).numpy()
		return torch.stack(all_embeddings)
	def predict(self, prev_sentence: str, target_sentence: str) -> dict:
		"""SentenceBertを利用し、二つの文章の類似度を測定する。
		Args:
			prev_sentence (str): The sentence before last
			target_sentence (str): The sentence to be identified
		Returns:
			str: Json string containing the identification results
		"""

		encode_results = self.encode([prev_sentence, target_sentence])
		prev_sentence_enc, target_sentence_enc = encode_results[0], encode_results[1]
		score = (
			1
			- scipy.spatial.distance.cdist(
				np.reshape(prev_sentence_enc, (1, -1)),
				np.reshape(target_sentence_enc, (1, -1)),
				metric="cosine",
			)[0][0]
		)  # [0-1]の範囲
		return score if score >= 0 else 0

sentence_master = []

class SentenceInfo:
	def __init__(self,myIdx,sentence,dic):
		self.sentence = sentence
		self.eval_dic = dic
		self.idx = myIdx
		self.top_n = 5
		self.similarity_top_n = []
	def addEvaluation(self, idx, eval):
		self.eval_dic[idx] = eval
		if len(self.eval_dic) >= self.top_n:
			if len(self.similarity_top_n) == 0:
				self.setTopEvalDic(self.eval_dic)
			else:
				self.replaceTopEvalDic(idx,eval)
	def setTopEvalDic(self, eval_dic):
		sort_orders = sorted(eval_dic.items(), key=lambda x: x[1], reverse=True)
		for i in sort_orders:
			self.similarity_top_n.append([i[0],i[1]])
			print(i[0], i[1])
		self.similarity_top_n = sorted(self.similarity_top_n, reverse=True, key=lambda x: x[1])
		self.similarity_top_n = self.similarity_top_n[:self.top_n]
	def replaceTopEvalDic(self, idx,eval):
		print("len = " + str(len(self.similarity_top_n)))
		if self.similarity_top_n[self.top_n - 1][1] < eval:
			#org_list = str(self.similarity_top_n)
			self.similarity_top_n[self.top_n - 1] = [idx,eval]
			self.similarity_top_n = sorted(self.similarity_top_n, reverse=True, key=lambda x: x[1])
			#print("replace:" + org_list + " -> " + str(self.similarity_top_n))
	def getSentence(self):
		return self.sentence
	def getIdx(self):
		return self.idx
	def getEvalDictionary(self):
		return self.eval_dic
		

class Classify:
	def __init__(self,type):
		self.dic = {}
		self.sentences = []
		self.sbj = SentenceBertJapanese("sonoisa/sentence-bert-base-ja-mean-tokens")
		self.type = type
	def setType(self,type):
		self.type = type
	def addKeyword(self,keyword):
		self.dic[keyword] = 1
	def evaluateKeyword(self,sentence):
		return false
	def evaluateSentence(self,sentence):
		idx = len(sentence_master)
		idx_dic = {}
		if idx != 0:
			for target in self.sentences:
				eval = self.sbj.predict(sentence, target.getSentence())
				# この文章用の比較結果蓄積
				idx_dic[target.getIdx()] = eval
				# 比較対象にこの文章との比較結果を追加
				target.addEvaluation(idx, eval)
		# sentenceに関する情報を追加
		sentence_master.append(sentence)
		self.sentences.append(SentenceInfo(idx,sentence,idx_dic))
	def showSentenceInfo(self,sentenceInfo):
		print(sentenceInfo.getSentence())
		print(str(sentenceInfo.getEvalDictionary()))
	def showAllSentenceInfo(self):
		for sentence in self.sentences:
			self.showSentenceInfo(sentence)
	def createWriteBuffer(self):
		wlist = []
		for target in self.sentences:
			wbuff = target.getSentence() + "," + str(self.type) + "\n"
			wlist.append(wbuff)
		return wlist
	def readFile(self,fname):
		with open(fname) as f:
			for l in f.readlines():
				ll = l.split(",")
				sentence = ll[0]
				type = int(ll[1].strip())
				if type == self.type:
					self.evaluateSentence(sentence)


class SituationClassifier(Classify):
	def __init__(self):
		super().__init__(1)

class DecisionClassifier(Classify):
	def __init__(self):
		super().__init__(2)

class IssueClassifier(Classify):
	def __init__(self):
		super().__init__(3)


#class SentenceAnalyzer:
#	pass


if __name__ == "__main__":
	import time

	start = time.time()

	checkpoint = time.time()
	print("モデルの読み込み時間: {}s".format(time.time() - start))
	c = SituationClassifier()
	c.readFile("test_i.csv")
	c.showAllSentenceInfo()
	buff = c.createWriteBuffer()
	file = open("test_o.csv","w", encoding = "utf_8")
	file.writelines(buff)
	file.close()
	print("推論時間: {}s".format(time.time() - checkpoint))
