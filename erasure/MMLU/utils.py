TASKS = [
        'abstract_algebra',
        'anatomy',
        'astronomy',
        'business_ethics',
        'clinical_knowledge',
        'college_biology',
        'college_chemistry',
        'college_computer_science',
        'college_mathematics',
        'college_medicine',
        'college_physics',
        'computer_security',
        'conceptual_physics',
        'econometrics',
        'electrical_engineering',
        'elementary_mathematics',
        'formal_logic',
        'global_facts',
        'high_school_biology',
        'high_school_chemistry',
        'high_school_computer_science',
        'high_school_european_history',
        'high_school_geography',
        'high_school_government_and_politics',
        'high_school_macroeconomics',
        'high_school_mathematics',
        'high_school_microeconomics',
        'high_school_physics',
        'high_school_psychology',
        'high_school_statistics',
        'high_school_us_history',
        'high_school_world_history',
        'human_aging',
        'human_sexuality',
        'international_law',
        'jurisprudence',
        'logical_fallacies',
        'machine_learning',
        'management',
        'marketing',
        'medical_genetics',
        'miscellaneous',
        'moral_disputes',
        'moral_scenarios',
        'nutrition',
        'philosophy',
        'prehistory',
        'professional_accounting',
        'professional_law',
        'professional_medicine',
        'professional_psychology',
        'public_relations',
        'security_studies', 
        'sociology',
        'us_foreign_policy',
        'virology',
        'world_religions']

BIOLOGY_TASKS = ['anatomy',
                 'clinical_knowledge',
                 'college_biology',
                 'college_medicine',
                 'high_school_biology',
                 'human_aging',
                 'medical_genetics',
                 'nutrition',
                 'professional_medicine',
                 'virology']

OTHER_TASKS = [task for task in TASKS if task not in BIOLOGY_TASKS]


def llama_chat_prompt(prompt, system_message=None):
    if system_message == None:
        return f"[INST]\n{prompt} [/INST]"
    return f"[INST]\n<<SYS>>\n{system_message}\n<</SYS>>\n{prompt} [/INST]"

def test_answer_mmlu_(pred_str, ans):
    pattern = 'the answer is ('
    pred = pred_str.lower().split(pattern)
    
    if(len(pred) > 1):
        # print(pred)
        pred = pred[1][0]
        gold = ans.lower()
        # print('debug 1, pred %s, gold %s' % (pred, gold))
        return pred == gold
    else: 
        pred = 'C'
        # print(ans_str)
        gold = ans.lower()
        # print('debug 2, pred %s, gold %s' % (pred, gold))
        return pred == gold

# extract answer in pred_str and compare with ans_str
def test_answer_mmlu_claude_instant(pred_str, ans_str):
    pattern = 'the answer is '
    pred = pred_str.lower().split(pattern)
    if len(pred) == 1:
        return False
    else:
        return pred[1][0] == ans_str.lower()

def test_answer_mmlu_claude(pred_str, ans_str):
    pattern = 'the answer is '
    pred = pred_str.lower().split(pattern)
    
    if(len(pred) > 1):
        # print(pred)
        pred = pred[1]
        for p in pred: 
            if(p.isalpha()): break
        pred = p
        print(ans_str)
        gold = ans_str.lower()
        print('debug 1, pred %s, gold %s' % (pred, gold))
        return pred == gold
    else: 
        pred = 'c'
        # print(ans_str)
        gold = ans_str.lower()
        # print('debug 2, pred %s, gold %s' % (pred, gold))
        return pred == gold

def test_answer_mmlu(pred_str, ans_str):
    pattern = 'the answer is ('
    pred = pred_str.lower().split(pattern)
    
    if(len(pred) > 1):
        # print(pred)
        pred = pred[1][0]
        gold = ans_str.split('A:\n')[1][0].lower()
        # print('debug 1, pred %s, gold %s' % (pred, gold))
        return pred == gold
    else: 
        pred = 'C'
        # print(ans_str)
        gold = ans_str.split('A:\n')[1][0].lower()
        # print('debug 2, pred %s, gold %s' % (pred, gold))
        return pred == gold

def parse_pred_ans(filename):
    with open(filename) as fd: lines = fd.readlines()
    am, a = None, None
    num_q, acc = 0, 0
    current_mode = 'none'
    questions = []
    ans_pred = []
    ans_gold = []
    for l in lines:
        if(l.startswith('Q: ')):
            if(am is not None and a is not None):
                questions.append(q)
                ans_pred.append(am)
                ans_gold.append(a)
                # print(am)
                # print(a)
                if(test_answer_mmlu(am, a)):
                    acc += 1
            current_mode = 'q'
            q = l
            num_q += 1
        elif(l.startswith('A_model:')):
            current_mode = 'am'
            am = l
        elif(l.startswith('A:') and not l.startswith("A: Let's think step by step")):
            current_mode = 'a'
            a = l
        else:
            if(current_mode == 'q'): q += l
            elif(current_mode == 'am'): am += l
            elif(current_mode == 'a'): a += l
            else:
                raise ValueError(current_mode)
                
    questions.append(q)
    ans_pred.append(am)
    ans_gold.append(a)
    # print(am)
    # print(a)
    if(test_answer_mmlu(am, a)):
        acc += 1
    print('num_q %d correct %d ratio %.4f' % (num_q, acc, float(acc / num_q)))
    return questions, ans_pred, ans_gold

def test_finished(ans_model):
    if('answer is' in ans_model): return True
    else: return False

def extract_ans(ans_model):
    ans_model = ans_model.split('\n')
    ans = []
    residual = []
    for li, al in enumerate(ans_model):
        ans.append(al)
        if('answer is' in al):
            break
    residual = list(ans_model[li + 1:])
    ans = '\n'.join(ans)
    residual = '\n'.join(residual)
    return ans, residual
