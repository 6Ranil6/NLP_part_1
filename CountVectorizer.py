from typing import List
from pprint import pprint

class CountVectorizer:

    def __init__(self):
        self.vocabulary = dict()

    def _transform_sentence(self, sen: str) -> str:
        """Нормализация предложения

        Args:
            sen (str): предложение

        Returns:
            str: нормализованное предложение
        """
        return [token.lower() for token in sen.split()]

    def __compute_vocabular(self, all_words: List[str]) -> None:
        """Создает словарь

        Args:
            all_words (List[str]): все слова в документе/ корпусе
        """
        uniq_words = []
        for el in all_words:
            if el not in uniq_words:
                uniq_words.append(el)

        for idx, word in enumerate(uniq_words):
            self.vocabulary[word] = idx

    def fit_transform(self, corpus: List[str]) -> List[List[int]]:
        """Обучение векторайзера и преобразование всего корпаса в матрицу

        Args:
            corpus (List[str]): документ со всеми предложениями

        Returns:
            List[int]: матрица с подсчетом слов в предложении
        """
        new_corpus = [self._transform_sentence(sen) for sen in corpus]
        merge_lists = lambda main_list: [el for line in main_list for el in line]
        self.__compute_vocabular(merge_lists(new_corpus))

        matrix = [[0] * len(self.vocabulary) for _ in range(len(corpus))]
        for i, sentence in enumerate(new_corpus):
            for word in sentence:
                matrix[i][self.vocabulary[word]] += 1

        return matrix

    def transform(self, text: str) -> List[int]:
        """
        Преобразование предложения в матрицу
        Args:
            text (str): предложение

        Returns:
            List[int]: вектор с подсчетом слов в предложении
        """
        vector = [0] * len(self.vocabulary)
        new_text = self._transform_sentence(text)
        for word in new_text:
                vector[self.vocabulary[word]] += 1
        return vector

    def get_feature_names(self) -> List[str]:
        """Возвращает слова в свое словаре

        Returns:
            List[str]: список слов
        """
        return list(self.vocabulary.keys())

def main():
    corpus = [
        "Crock Pot Pasta Never boil pasta again",
        "Pasta Pomodoro Fresh ingredients Parmesan to taste",
    ]
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)
    pprint(vectorizer.get_feature_names())
    pprint(count_matrix)
    pprint(vectorizer.transform("Pasta Pomodoro Fresh"))

if __name__ == "__main__":
    main()
