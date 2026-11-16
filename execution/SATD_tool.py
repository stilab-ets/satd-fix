import difflib
import re
import json
import os
import ast
import black
import Levenshtein
from crystalbleu import corpus_bleu
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import difflib


def find_files_with_prefix(path, start_file_name):
    """Return files whose names start with the supplied prefix."""
    matching_files = []
    for file_name in os.listdir(path):
        if file_name.startswith(start_file_name):
            matching_files.append(os.path.join(path, file_name))
    return matching_files

def save_to_json(data, filename):
    if '/' in filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

def load_from_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_from_jsons(filenames):
    data = {}
    for filename in filenames:
        data.update(load_from_json(filename))
    return data


def get_deleted_and_inserted_lines(oldcode, newcode, ignore_whitespace=True):
    """Return nonblank deleted and inserted lines, optionally ignoring whitespace."""
    old_lines = oldcode.splitlines()
    new_lines = newcode.splitlines()
    if ignore_whitespace:
        old_lines = [re.sub(r'\s+', '', line) for line in old_lines]
        new_lines = [re.sub(r'\s+', '', line) for line in new_lines]
    differ = difflib.Differ()
    diff = list(differ.compare(old_lines, new_lines))
    deleted = []
    inserted = []
    for line in diff:
        if line.startswith("- ") and line[2:].strip()!='':
            deleted.append(line[2:].strip())  
        elif line.startswith("+ ") and line[2:].strip()!='':
            inserted.append(line[2:].strip())  
    return deleted, inserted

def extract_code(text, language, dataset=None):
    """Extract a code block or method from the model response."""
    code = get_largest_code_section(text)
    if code != '':
        return code
    lines = text.split('\n')
    start_line = None
    for i,line in enumerate(lines):
        if '```' in line:
            start_line = i+1
            break
        if language.lower()=='python':
            if line.strip().startswith('def ') or line.strip().startswith('class '):
                start_line = i
                break
        elif language.lower()=='java':
            if len(line.strip().split())>0 and line.strip().split()[0] in ('public', 'private', 'protected', 'class', 'interface', 'void', 'int', 'String', 'boolean', 'double', 'float', 'char'):
                if dataset=='Mastropaolo' and i>0 and lines[i-1].startswith('@'):
                    start_line = i-1
                else:
                    start_line = i
                break
        else:
            raise ValueError('Language not supported')
    if start_line is not None:
        end_line = len(lines)
        for i, line in enumerate(lines):
            if i > start_line:
                if '```' in line or (line and line[0].isupper()):
                    end_line = i
                    break
        return '\n'.join(lines[start_line:end_line])
    return ''

def get_largest_code_section(text):
    """Return the largest fenced code section in the response."""
    lines = text.split('\n')
    start_lines = []
    end_lines = []
    for i,line in enumerate(lines):
        if '```' in line:
            if len(start_lines)==len(end_lines):
                start_lines.append(i+1)
            else:
                end_lines.append(i)
    largest_index = None
    for i in range(len(end_lines)):
        if largest_index==None or end_lines[i]-start_lines[i] > end_lines[largest_index]-start_lines[largest_index]:
            largest_index=i
    if largest_index is not None:
        return '\n'.join(lines[start_lines[largest_index]:end_lines[largest_index]])
    else:
        return ""
    
def remove_docstrings_and_comments_by_ast_from_python(code):
    tree = ast.parse(code)
    class RemoveComments(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return None  
            return node
    tree = RemoveComments().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)

def remove_docstrings_and_comments_by_regex_from_python(code):
    single_line_comment_pattern = r'(?<!\\)#.*'
    multiline_comment_pattern = r'(\'\'\'|\"\"\").*?\1'
    code = re.sub(multiline_comment_pattern, '', code, flags=re.DOTALL)
    code = re.sub(single_line_comment_pattern, '', code)
    lines = code.splitlines()
    clean_lines = [line for line in lines if line.strip()]
    return "\n".join(clean_lines)

def remove_docstrings_and_comments_from_python(code):
    try:
        return remove_docstrings_and_comments_by_ast_from_python(code)
    except:
        return remove_docstrings_and_comments_by_regex_from_python(code)

def remove_imports(code):
    output = []
    for line in code.split('\n'):
        if line.lstrip().startswith('import ') or line.lstrip().startswith('from '):
            continue
        output.append(line)
    return '\n'.join(output)

def remove_comments_and_javadoc_from_java(code):
    code = re.sub(r'//.*', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'\n\s*\n', '\n', code)
    return code


def exact_match(code1, code2, language, ignore_docstrings_and_comments=False, ignore_whitespace=False, format_code=True):
    """Return edit distance after applying the selected code normalization."""
    def remove_empty_lines(code):
        return "\n".join([line for line in code.splitlines() if line.strip() != ""])
    if ignore_docstrings_and_comments:
        if language.lower()=='python':
            try: 
                code1 = remove_docstrings_and_comments_by_ast_from_python(code1)
                code2 = remove_docstrings_and_comments_by_ast_from_python(code2)
            except:
                code1 = remove_docstrings_and_comments_by_regex_from_python(code1)
                code2 = remove_docstrings_and_comments_by_regex_from_python(code2)
        elif language.lower()=='java':
            code1 = remove_comments_and_javadoc_from_java(code1)
            code2 = remove_comments_and_javadoc_from_java(code2)
        else:
            raise ValueError('Language not supported')
        code1 = remove_imports(code1)
        code2 = remove_imports(code2)
    if ignore_whitespace:
        processed_code1 = re.sub(r'\s+', '', code1)
        processed_code2 = re.sub(r'\s+', '', code2)
    elif format_code:
        if language.lower()=='python':
            processed_code1 = black.format_str(code1, mode=black.Mode())
            processed_code2 = black.format_str(code2, mode=black.Mode())
            processed_code1 = remove_empty_lines(processed_code1)
            processed_code2 = remove_empty_lines(processed_code2)
        elif language.lower()=='java':
            processed_code1 = re.sub(r'\s+', '', code1)
            processed_code2 = re.sub(r'\s+', '', code2)
        else:
            raise ValueError('Language not supported')
    else:
        processed_code1 = code1
        processed_code2 = code2
    return Levenshtein.distance(processed_code1, processed_code2)

def get_exact_matches(df, randIndex_answer, language, dataset, extract_code_from_answer, ignore_docstrings_and_comments=False, ignore_whitespace=False, format_code=True):
    em = {}
    randIndex_distance = {}
    black_failed = []
    black_failed_em = {}
    for i in randIndex_answer:
        if extract_code_from_answer:
            code1 = extract_code(randIndex_answer[i], language, dataset)
        else:
            code1 = randIndex_answer[i]
        code2 = df[df['rand_index']==i]['containing_method_after_repayment'].tolist()[0]
        try:
            distance = exact_match(code1, code2, language, ignore_docstrings_and_comments=ignore_docstrings_and_comments, ignore_whitespace=ignore_whitespace, format_code=format_code)
            randIndex_distance[i] = distance
            if distance==0:
                em[i] = extract_code(randIndex_answer[i], language, dataset)
        except:
            black_failed.append(i)
            distance = exact_match(code1, code2, language, ignore_docstrings_and_comments=ignore_docstrings_and_comments, ignore_whitespace=True)
            randIndex_distance[i] = distance
            if distance==0:
                em[i] = extract_code(randIndex_answer[i], language, dataset)
                black_failed_em[i] = extract_code(randIndex_answer[i], language, dataset)
    return randIndex_distance, em, black_failed, black_failed_em


def get_itmes_having_specific_number_of_inserted_lines(df,randIndex,min_inserted_lines,max_inserted_lines):
    items = []
    for i in randIndex:
        inserted_lines = df[df['rand_index']==i]['number_of_inserted_lines'].tolist()[0]
        if min_inserted_lines<=inserted_lines<=max_inserted_lines:
            items.append(i)
    return items


TOKENIZATION_APPROACH = 're' 

def get_tokens(text):
    if TOKENIZATION_APPROACH=='simple':
        return text.split()
    elif TOKENIZATION_APPROACH=='re':
        return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    else:
        raise ValueError('Tokenization approach not supported:', TOKENIZATION_APPROACH)

        
def get_BLEU(randIndex_reference, randIndex_candidate, language, remove_icd):
    """Calculate token-level BLEU for each candidate repair."""
    BLEU = {}
    for i in randIndex_candidate:
        if remove_icd:
            if language=='Python':
                try:
                    references = [get_tokens(remove_imports(remove_docstrings_and_comments_by_ast_from_python(randIndex_reference[i])))]
                    candidates = get_tokens(remove_imports(remove_docstrings_and_comments_by_ast_from_python(randIndex_candidate[i])))
                except:
                    references = [get_tokens(remove_imports(remove_docstrings_and_comments_by_regex_from_python(randIndex_reference[i])))]
                    candidates = get_tokens(remove_imports(remove_docstrings_and_comments_by_regex_from_python(randIndex_candidate[i])))
            elif language=='Java':
                references = [get_tokens(remove_imports(remove_comments_and_javadoc_from_java(randIndex_reference[i])))]
                candidates = get_tokens(remove_imports(remove_comments_and_javadoc_from_java(randIndex_candidate[i])))
        else:            
            references = [get_tokens(randIndex_reference[i])]
            candidates = get_tokens(randIndex_candidate[i])
        BLEU[i] = sentence_bleu(references, candidates, smoothing_function=SmoothingFunction().method4)
    return BLEU

def get_crystalBLEU(randIndex_reference, randIndex_candidate, language, remove_icd, trivially_shared_ngrams):
    """Calculate CrystalBLEU while excluding shared n-grams."""
    crystalBLEU = {}
    for i in randIndex_candidate:
        if remove_icd:
            if language=='Python':
                try:
                    references = [[remove_imports(remove_docstrings_and_comments_by_ast_from_python(randIndex_reference[i]))]]
                    candidates = [remove_imports(remove_docstrings_and_comments_by_ast_from_python(randIndex_candidate[i]))]
                except:
                    references = [[remove_imports(remove_docstrings_and_comments_by_regex_from_python(randIndex_reference[i]))]]
                    candidates = [remove_imports(remove_docstrings_and_comments_by_regex_from_python(randIndex_candidate[i]))]
            elif language=='Java':
                references = [[remove_imports(remove_comments_and_javadoc_from_java(randIndex_reference[i]))]]
                candidates = [remove_imports(remove_comments_and_javadoc_from_java(randIndex_candidate[i]))]
        else:
            references = [[randIndex_reference[i]]]
            candidates = [randIndex_candidate[i]]
        crystalBLEU[i] = corpus_bleu(references, candidates, ignoring=trivially_shared_ngrams)
    return crystalBLEU


def get_linePRF(randIndex_reference, randIndex_candidate, language, remove_icd, ignore_whitespace=True):
    """Calculate line precision, recall, and F1 for each repair."""
    lineP = {}
    lineR = {}
    lineF = {}
    for i in randIndex_candidate:
        if remove_icd:
            if language=='Python':
                try:
                    references = remove_imports(remove_docstrings_and_comments_by_ast_from_python(randIndex_reference[i]))
                    candidates = remove_imports(remove_docstrings_and_comments_by_ast_from_python(randIndex_candidate[i]))
                except:
                    references = remove_imports(remove_docstrings_and_comments_by_regex_from_python(randIndex_reference[i]))
                    candidates = remove_imports(remove_docstrings_and_comments_by_regex_from_python(randIndex_candidate[i]))
            elif language=='Java':
                references = remove_imports(remove_comments_and_javadoc_from_java(randIndex_reference[i]))
                candidates = remove_imports(remove_comments_and_javadoc_from_java(randIndex_candidate[i]))
        else:
            references = randIndex_reference[i]
            candidates = randIndex_candidate[i]
        if language=='Java': 
            ref_lines = set([line.strip() for line in references.split('\n') if line.strip()!=''])
            can_lines = set([line.strip() for line in candidates.split('\n') if line.strip()!=''])
        else:
            ref_lines = set([line.strip() for line in references.split('\n') if line.strip()!=''])
            can_lines = set([line.strip() for line in candidates.split('\n') if line.strip()!=''])
        if ignore_whitespace:
            ref_lines = set([re.sub(r'\s+', '', line) for line in ref_lines])
            can_lines = set([re.sub(r'\s+', '', line) for line in can_lines])
        if len(ref_lines) > 0 and len(can_lines) > 0:
            lineP[i] = len(can_lines & ref_lines) / len(can_lines)
            lineR[i] = len(can_lines & ref_lines) / len(ref_lines)
            if (lineP[i] + lineR[i]) > 0:
                lineF[i] = 2 * (lineP[i] * lineR[i]) / (lineP[i] + lineR[i])
            else:
                lineF[i] = 0
        else:
            lineP[i] = 0
            lineR[i] = 0
            lineF[i] = 0
    return lineP, lineR, lineF


def get_updated_or_new_lines(oldcode: str, newcode: str, dataset) -> list:
    """Return inserted and deleted lines for change-based similarity metrics."""
    oldlines = oldcode.splitlines()
    newlines = newcode.splitlines()

    if dataset=='Mastropaolo':
        oldlines = [l.strip() for l in oldlines if l.strip()]
        newlines = [l.strip() for l in newlines if l.strip()]
    
    if False:
        oldlines = [re.sub(r'\s+', ' ', l) for l in oldlines]
        newlines = [re.sub(r'\s+', ' ', l) for l in newlines]

    diff = difflib.ndiff(oldlines, newlines)

    updated_or_new_lines = [line[2:] for line in diff if line.startswith(('+ ', '- '))] 

    return updated_or_new_lines

