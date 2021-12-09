# This script contains data loading and processing class

from fastNLP import Vocabulary
from fastNLP import DataSet, Instance
from fastNLP.io import Loader, Pipe, DataBundle


def bioes2bio(tags):
    """ Transform the encoding type from `bioes` to `bio` """
    new_tags = []
    for tag in tags:
        if tag.startswith('S'):
            tag = 'B' + tag[1:]
        if tag.startswith('E'):
            tag = 'I' + tag[1:]
        new_tags.append(tag)
    return new_tags


def bioes2to(tags):
    """ Transform the encoding type from `bioes` to `to` """
    new_tags = []
    for tag in tags:
        if tag != 'O':
            tag = 'T' + tag[1:]
        new_tags.append(tag)
    return new_tags


def get_ate(tags):
    """ 
    Given a tag seq, extract the target seq for ATE task
    """
    ate_tags = []
    for tag in tags:
        if tag == 'O':
            ate_tags.append(tag)
        else:
            ate_tags.append(tag[0])
    return ate_tags


def get_asc(tags):
    """ 
    Given a tag seq, extract the target seq for ASC task
    """
    asc_tags = []
    for tag in tags:
        if tag == 'O':
            asc_tags.append(tag)
        else:
            asc_tags.append(tag[2:])
    return asc_tags


class ABSAQALoader(Loader):
    """ 
    Load ABSA-QA data saved in Conll data format
    """
    def __init__(self, qa_match=False):
        super().__init__()
        self.qa_match = qa_match

    def _load(self, path=str):
        """
        Given a data `path` of train/dev/test, return `Dataset`
        The data is written in conll data format: each line is: "word tag"
        """
        dataset = DataSet()
        with open(path, 'r', encoding='utf-8') as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if line:
                    word, tag = line.split('\t')
                    words.append(word)
                    tags.append(tag)
                else:
                    assert len(words) == len(tags)
                    sep_idx = words.index('[QA_SEP]')
                    
                    # read in data in specified format
                    # read qa as two separate seqs
                    raw_q = words[:sep_idx]
                    raw_a = words[sep_idx+1:]
                    aspect_seq = tags[:sep_idx]
                    if not self.qa_match:
                        dataset.append(
                            Instance(raw_q=raw_q, raw_a=raw_a, target=aspect_seq)
                        )
                    else:
                        dataset.append(
                            Instance(raw_q=raw_q, raw_a=raw_a, target=tags[sep_idx])
                        )
                    words, tags = [], []
  
        return dataset


def separate_unfied_labels(data_bundle, old_field='target'):
    """ 
    Separate the unfied tag sequence to two sequences 
    """
    for name, dataset in data_bundle.datasets.items():
        # keep a copy of the original unfied target seq
        dataset.copy_field(field_name='target', new_field_name='unified_target')
        
        # 'B-POS' -> 'B', denoting aspect term extraction
        dataset.apply_field(get_ate, field_name=old_field, new_field_name='ate_target')
        # 'B-POS' -> 'POS', denoting sentiment classification
        dataset.apply_field(get_asc, field_name=old_field, new_field_name='asc_target')
        
        # delete `target` fields to avoid confusion in later processing
        dataset.delete_field('target')
    
    return data_bundle


def indexize_input_field(data_bundle, input_fields):
    """ 
    Indexize the input data fields
    """
    word_vocab = Vocabulary()
    word_vocab.from_dataset(*[ds for name, ds in data_bundle.datasets.items() if 'train' in name],
                            field_name=input_fields,
                            no_create_entry_dataset=[ds for name, ds in data_bundle.datasets.items() 
                                                        if 'train' not in name])
    word_vocab.index_dataset(*data_bundle.datasets.values(), field_name=input_fields)
    data_bundle.set_vocab(word_vocab, 'words')

    return data_bundle


def indexize_output_field(data_bundle, output_fields):
    """ 
    Indexize the output data fields 
    """
    # if we only want the `unified` target seq
    if isinstance(output_fields, str):
        output_fields = [output_fields]
    
    # otherwise each target seq has one vocab
    for target_field_name in output_fields:
        tgt_vocab = Vocabulary(unknown=None, padding=None)
        tgt_vocab.from_dataset(*[ds for name, ds in data_bundle.iter_datasets() if 'train' in name],
                               field_name=target_field_name,
                               no_create_entry_dataset=[ds for name, ds in data_bundle.iter_datasets()
                                                        if ('train' not in name) and (ds.has_field(target_field_name))]
                               )
        
        tgt_vocab.index_dataset(*[ds for ds in data_bundle.datasets.values() if ds.has_field(target_field_name)], field_name=target_field_name)
        data_bundle.set_vocab(tgt_vocab, target_field_name)

    return data_bundle


class ABSAQAPipe(Pipe):
    """
    A pipe class for pre-processing the ABSA-QA dataset
    """
    def __init__(self, encoding_type='bioes', output_format='separate'):
        super().__init__()
        
        # check tagging scheme
        if encoding_type == 'bioes':
            self.encoding_func = lambda x: x
        elif encoding_type == 'bio':
            self.encoding_func = bioes2bio
        elif encoding_type == 'to':
            self.encoding_func = bioes2to
        else:
            raise Exception(f"Doesn't support `{encoding_type}` encoding type!")      
    
        # check output format
        if output_format in ['unified', 'separate']:
            self.output_format = output_format
        else:
            raise Exception("Pls check the output tag format")

    def process(self, data_bundle: DataBundle) -> DataBundle:
        
        # process output tag seq format
        data_bundle.apply_field(self.encoding_func, field_name='target', new_field_name='target')
        if self.output_format == 'separate':
            data_bundle = separate_unfied_labels(data_bundle, 'target')
            target_fields = ['ate_target', 'asc_target', 'unified_target']
        else:
            target_fields = 'target'

        indexize_output_field(data_bundle, target_fields)

        # treat Q and A as two seqs but sharing the same vocab
        data_bundle.rename_field(field_name='raw_q', new_field_name='question', ignore_miss_dataset=True)
        data_bundle.rename_field(field_name='raw_a', new_field_name='answer', ignore_miss_dataset=True)
        input_fields = ['question', 'answer']
        indexize_input_field(data_bundle, input_fields)
        seq_len_fields = ['q_len', 'a_len']
        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len('question', 'q_len')
            dataset.add_seq_len('answer', 'a_len')
        input_fields = input_fields + seq_len_fields

        if isinstance(target_fields, str):
            target_fields = [target_fields]
        
        input_fields = input_fields + target_fields
        target_fields = target_fields + seq_len_fields

        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)

        return data_bundle

    def process_from_file(self, paths):
        """ 
        Use the corresponding loader (ABSAQALoader) to load the data 
        """
        loader = ABSAQALoader()
        data_bundle = loader.load(paths)
        return self.process(data_bundle)


class QAMatchPipe(Pipe):
    """
    A pipe class for the QA matching task
    """
    def __init__(self):
        super().__init__()
        
    def process(self, data_bundle: DataBundle) -> DataBundle:
        
        indexize_output_field(data_bundle, 'target')

        #treat Q and A as two seqs but sharing the same vocab
        data_bundle.rename_field(field_name='raw_q', new_field_name='question', ignore_miss_dataset=True)
        data_bundle.rename_field(field_name='raw_a', new_field_name='answer', ignore_miss_dataset=True)
        input_fields = ['question', 'answer']
        indexize_input_field(data_bundle, input_fields)
        seq_len_fields = ['q_len', 'a_len']
        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len('question', 'q_len')
            dataset.add_seq_len('answer', 'a_len')
        input_fields = input_fields + seq_len_fields
        
        input_fields = input_fields + ['target']
        target_fields = ['target'] + seq_len_fields

        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)

        return data_bundle
    
    def process_from_file(self, paths):
        """ 
        Re-Use the corresponding loader (ABSAQALoader) to load the data 
        """
        loader = ABSAQALoader(qa_match=True)
        data_bundle = loader.load(paths)
        return self.process(data_bundle)
