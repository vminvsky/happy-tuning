import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_training_format(body, parent_body):
    # TODO consider better structures for training
    # important to discriminate between body and parent.
    # since we'll have a fixed training size (?), I'll put the repsonse first
    return f"Comment response: {body}\n\nMain message: {parent_body}"

def process(file_path):
    df = pd.read_csv(file_path)
    topics = df.columns[5:]

    df["text"] = df.apply(lambda x: prepare_training_format(x["comment_body"], x["parent_body"]), axis=1)
    
    
    df["labels"] = df[topics].values.tolist()
    df = df[["id", "text", "labels"]]
    
    # train test split
    df_train, df_test = train_test_split(df, test_size=0.1)
    df_train.to_json("train.json", orient="records", indent=2)
    df_test.to_json("test.json", orient="records", indent=2)
    
if __name__=="__main__":
    path = "9998_sample_topics.csv"
    process(path)
    
    