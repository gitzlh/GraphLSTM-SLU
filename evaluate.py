import numpy as np


# compute f1 score is modified from conlleval.pl
def __startOfChunk(prevTag, tag, prevTagType, tagType, chunkStart=False):
    if prevTag == 'B' and tag == 'B':
        chunkStart = True
    if prevTag == 'I' and tag == 'B':
        chunkStart = True
    if prevTag == 'O' and tag == 'B':
        chunkStart = True
    if tag != 'O' and tag != '.' and prevTagType != tagType:
        chunkStart = True
    return chunkStart


def __endOfChunk(prevTag, tag, prevTagType, tagType, chunkEnd=False):
    if prevTag == 'B' and tag == 'B':
        chunkEnd = True
    if prevTag == 'B' and tag == 'O':
        chunkEnd = True
    if prevTag == 'I' and tag == 'B':
        chunkEnd = True
    if prevTag == 'I' and tag == 'O':
        chunkEnd = True
    if prevTag != 'O' and prevTag != '.' and prevTagType != tagType:
        chunkEnd = True
    return chunkEnd


def __splitTagType(tag):
    s = tag.split('-')
    if len(s) > 2 or len(s) == 0:
        raise ValueError('tag format wrong. it must be B-xxx.xxx')
    if len(s) == 1:
        tag = s[0]
        tagType = ""
    else:
        tag = s[0]
        tagType = s[1]
    return tag, tagType


def computeF1Score(correct_slots, pred_slots):
    '''
    :param correct_slots:[[],[],[]]
    :param pred_slots: [[],[],[]]
    :return:
    '''
    correctChunk = {}
    correctChunkCnt = 0
    foundCorrect = {}
    foundCorrectCnt = 0
    foundPred = {}
    foundPredCnt = 0
    correctTags = 0
    tokenCount = 0
    for correct_slot, pred_slot in zip(correct_slots, pred_slots):
        # print('correct_slot:',len(correct_slot))
        # print('pred_slot:',len(pred_slot))
        correct = False
        lastCorrectTag = 'O'  # o not 0
        lastCorrectType = ''
        lastPredTag = 'O'
        lastPredType = ''

        lastCorrectChunkCnt = correctChunkCnt
        for c, p in zip(correct_slot, pred_slot):
            correctTag, correctType = __splitTagType(c)
            predTag, predType = __splitTagType(p)

            if correct == True:  # until we can be sure it is correct,it's incorrect
                if __endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and __endOfChunk(
                        lastPredTag, predTag, lastPredType,
                        predType) == True and (
                        lastCorrectType == lastPredType):  # we can finally say it's correct
                    correct = False
                    correctChunkCnt += 1
                    if lastCorrectType in correctChunk:
                        correctChunk[lastCorrectType] += 1
                    else:
                        correctChunk[lastCorrectType] = 1
                elif __endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) != __endOfChunk(lastPredTag,
                                                                                                            predTag,
                                                                                                            lastPredType,
                                                                                                            predType) or (
                        correctType != predType):
                    correct = False

                    # if __name__ == '__main__':
                    # print('mistake:')
                    # print('correct_slot:', correct_slot)
                    # print('pred_slot:   ', pred_slot)
                    # print('\n')
            #
            if __startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and __startOfChunk(
                    lastPredTag, predTag, lastPredType,
                    predType) == True and (
                    correctType == predType):
                correct = True

            if __startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True:
                foundCorrectCnt += 1
                if correctType in foundCorrect:
                    foundCorrect[correctType] += 1
                else:
                    foundCorrect[correctType] = 1

            if __startOfChunk(lastPredTag, predTag, lastPredType, predType) == True:
                foundPredCnt += 1
                if predType in foundPred:
                    foundPred[predType] += 1
                else:
                    foundPred[predType] = 1

            if correctTag == predTag and correctType == predType:
                correctTags += 1
            tokenCount += 1
            lastCorrectTag = correctTag
            lastCorrectType = correctType
            lastPredTag = predTag
            lastPredType = predType

        if correct == True:
            correctChunkCnt += 1
            if lastCorrectType in correctChunk:
                correctChunk[lastCorrectType] += 1
            else:
                correctChunk[lastCorrectType] = 1

        else:
            pass

    # print('correctcnt:',correctChunkCnt)
    # print('foundpredcnt:',foundPredCnt)
    # print('foundcorrectcnt:',foundCorrectCnt)

    if foundPredCnt > 0:
        precision = 100 * correctChunkCnt / foundPredCnt
    else:
        precision = 0

    if foundCorrectCnt > 0:
        recall = 100 * correctChunkCnt / foundCorrectCnt
    else:
        recall = 0
    if (precision + recall) > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0

    return f1, precision, recall


def judge_intent(pred, gold):
    if '#' not in pred and '#' not in gold:
        return gold == pred
    elif '#' in gold and '#' in pred:
        return gold == pred
    else:
        if '#' in pred and '#' not in gold:
            pred = pred.split('#')
            if gold == pred[0] or gold == pred[1]:
                return True
            else:
                return False
        elif '#' in gold and '#' not in pred:
            gold = gold.split('#')
            if pred == gold[0] or pred == gold[1]:
                return True
            else:
                return False


def intent_acc(correct_slots, pred_slots, main_tag=False):
    totalCnt = 0
    correctCnt = 0
    for i, (correct, pred) in enumerate(zip(correct_slots, pred_slots)):
        if main_tag:
            if judge_intent((pred[0].lower()), correct[0].lower()):
                correctCnt += 1
            # else:
            #     print('i:', i)
            #     print('correct:', correct[0])
            #     print('pred:', pred[0])
        else:
            if correct.lower() == pred.lower():
                correctCnt += 1
        totalCnt += 1
    return correctCnt / totalCnt


def get_sent_acc(truth_file, pred_file):
    n_total = 0
    n_correct = 0
    intent_correct = 0
    with open(truth_file) as f_truth, open(pred_file) as f_pred:
        for i, (truth, pred) in enumerate(zip(f_truth, f_pred)):
            n_total += 1
            if pred.lower() == truth.lower():
                n_correct += 1
            if pred.split()[0].lower() == truth.split()[0].lower():
                intent_correct+=1
    try:
        acc = (n_correct / n_total) * 100
        intent_acc = (intent_correct/n_total)*100
    except:
        acc = 0
        intent_acc=0
    return intent_acc,acc


def evaluate(truth_file, pred_file):
    pred_slots = []
    correct_slots = []
    with open(pred_file, 'r') as pred:
        for line in pred:
            pred_slots.append(line.strip().split())
    with open(truth_file, 'r') as truth:
        for line in truth:
            correct_slots.append(line.strip().split())

    f1, precision, recall = computeF1Score(correct_slots, pred_slots)
    # acc = intent_acc(correct_slots, pred_slots, main_tag=True) * 100

    print('f1:%.3f' % f1)
    print('precision:%.3f' % precision)
    print('recall:%.3f' % recall)
    # print('intent:%.3f' % acc)
    return f1, precision, recall


