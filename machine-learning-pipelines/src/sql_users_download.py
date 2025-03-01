import os

import pandas as pd
from sqlalchemy import create_engine

from stage1_users import generate_combinated_file, generate_combinated_file_new_actions


def extract_user_train_from_sql(users: list, folders: list, rounds: int):
    i = 0
    engine = create_engine(
        'mysql+pymysql://admin:12345678@clashroomkike.cayqpljkaacf.eu-north-1.rds.amazonaws.com:3306/sessions')
    sessions = \
        pd.read_sql("SELECT table_name FROM information_schema.tables WHERE table_schema='sessions'", engine)[
            "TABLE_NAME"].tolist()
    for user in users:
        if not os.path.exists("data/users/stage-0/" + user):
            os.makedirs("data/users/stage-0/" + user)
        for _ in range(rounds):
            for folder in folders:
                if not os.path.exists("data/users/stage-0/" + user + "/" + folder):
                     os.makedirs("data/users/stage-0/" + user + "/" + folder)
                pd.read_sql(sessions[i], engine).to_csv(
                    "data/users/stage-0/" + user + "/" + folder + "/" + sessions[i].replace(":", "") + ".csv", index=False)
                i += 1



if __name__ == '__main__':
    users = ["user0"]
    folders = ["Sitting", "Hand up", "Writing", "Typing", "Drawing", "Eating", "Drinking", "Clapping", "Drawing",
               "Standing", "Walking"]
    extract_user_train_from_sql(users, folders, 1)
