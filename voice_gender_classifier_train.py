
def main():
    from classifiers.lstm_classifier import LSTMVoiceGenderClassifier

    vgc = LSTMVoiceGenderClassifier()
    vgc.fit()


if __name__ == '__main__':
    main()