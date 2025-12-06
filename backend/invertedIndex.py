# ============================================
# invertedIndex.py - Build Inverted Index
# ============================================
# This builds the inverted index from preprocessed documents.
# The inverted index maps each term to the documents it appears in.

def build_inverted_index(processed_docs):
    """
    Builds an inverted index from preprocessed documents.
    
    Input: {doc_id: [token1, token2, ...], ...}
    
    Output:
    - inverted_index: {term: {df: X, postings: {doc_id: tf, ...}}, ...}
    - doc_length: {doc_id: length, ...}
    - avg_doc_length: average number of tokens per document
    
    Example output:
    {
        "machine": {
            "df": 2,                    # appears in 2 documents
            "postings": {1: 3, 4: 2}    # doc 1 has it 3 times, doc 4 has it 2 times
        },
        "learning": {...}
    }
    """
    
    inverted_index = {}   # the main index we're building
    doc_length = {}       # stores length of each document (for BM25 later)

    # go through each document
    for doc_id, tokens in processed_docs.items():
        
        # save document length (number of tokens)
        doc_length[doc_id] = len(tokens)

        # count how many times each word appears in THIS document
        # e.g., {"machine": 3, "learning": 2}
        term_counts = {}
        for token in tokens:
            if token in term_counts:
                term_counts[token] += 1
            else:
                term_counts[token] = 1

        # now add these counts to the inverted index
        for term, tf in term_counts.items():
            
            # if we haven't seen this term before, create entry for it
            if term not in inverted_index:
                inverted_index[term] = {
                    "df": 0,          # document frequency (how many docs have this term)
                    "postings": {}    # {doc_id: tf, doc_id: tf, ...}
                }

            # add this document to the term's postings list
            inverted_index[term]["postings"][doc_id] = tf

            # increment df because this term appears in a new document
            inverted_index[term]["df"] += 1

    # calculate average document length (useful for BM25 ranking)
    total_tokens = sum(doc_length.values())
    number_of_docs = len(doc_length)
    avg_doc_length = total_tokens / number_of_docs if number_of_docs > 0 else 0

    return inverted_index, doc_length, avg_doc_length