"""
This program makes a data frame for the initial test of activity based on research paper located here:
https://arxiv.org/pdf/1809.04041.pdf
This project will have to train an ML model to create a data set for predictive analysis
"""
import csv
import json
import json_t as dm
import requests
import pandas as pd
import time


def get_Git_data(user_pass, url):
    """
    :param user_pass: A Dict of the user name and password for the GitHub API
    :param url:  The URL for the GitHuB api REST call
    :return:  returns a json object with the information from the GitHub API
    """
    index = 1
    finished = False
    data = []
    while not finished:
        url_index = url + str(index)
        r = requests.get(url_index, auth=(user_pass['Username'], user_pass['Password']))
        # print(r.json())
        # print(index)
        if r.json() != {'message': 'Server Error'}:
            if r.json() == []:
                finished = True
            else:
                data = data + r.json()
            index += 1
        time.sleep(1)

    return data

# Connect to the GitHub api with Username and Password specified in github_api_info.json
with open("github_api_info.json") as read_file:
    user_pass = json.load(read_file)

# Read in the ESEM - Dataset which is the name of 994 repos and their activity status (archived/FSE are inactive) and
# and active is active
Data_frame = {'Repository':[], 'Status':[]}
with open('ESEM - Dataset.csv', newline='')as esemdata:
    labeled_data = csv.DictReader(esemdata, delimiter=',')
    for x in labeled_data:
        Data_frame['Repository'] += [x['Repository']]
        Data_frame['Status'] += [x['Status']]

# sets the interval month and index for the query
months = 24
interval = 3
index = 1

# creates the URLS which are going to be used in the GitHub API
while index < len(Data_frame['Repository']):
    repository_split = Data_frame['Repository'][index].split('/')
    print(repository_split[0])
    commit_url = 'https://api.github.com/repos/'+ repository_split[0] + '/' + repository_split[1] + \
                 '/commits?per_page=100&page='
    forks_url = 'https://api.github.com/repos/'+ repository_split[0] + '/' + repository_split[1] + \
                 '/forks?per_page=100&page='
    issues_url = 'https://api.github.com/repos/'+ repository_split[0] + '/' + repository_split[1] + \
                 '/issues?state=all&per_page=100&page='
    pulls_url = 'https://api.github.com/repos/'+ repository_split[0] + '/' + repository_split[1] + \
                 '/pulls?state=all&per_page=100&page='
    user_url = 'https://api.github.com/users/' + repository_split[0] + '/repos?per_page=100&page='

# ==================================================================================================================== #
#  COMMIT DATA
# ==================================================================================================================== #
    print(index)
    data = get_Git_data(user_pass, commit_url)

    last_date = dm.get_last_date(data)
    dates = dm.get_list_of_dates(last_date,months, interval)
    commit_labels = dm.create_labels('commit', months, interval)
    data_points = dm.get_commit_data(dates, data, commit_labels)

    for label in commit_labels:
        Data_frame[label] = [data_points[label]] if label  not in Data_frame else Data_frame[label] + [
            data_points[label]]

    commit_days_labels = dm.create_labels('max_days_since', months, interval)
    data_points = dm.get_max_days_wo_commit(dates, data, commit_days_labels)

    for label in commit_days_labels:
        Data_frame[label] = [data_points[label]] if label not in Data_frame else Data_frame[label] + [
            data_points[label]]

    max_commit_labels = dm.create_labels('max_commit', months, interval)
    owner_commit_labels = dm.create_labels('owner_commit', months, interval)
    distinct_commit_labels = dm.create_labels('dis_commit', months, interval)
    new_commit_labels = dm.create_labels('new_commit', months, interval)

    data_points = dm.get_contrib(dates, data, max_commit_labels, distinct_commit_labels, owner_commit_labels,
                              new_commit_labels, repository_split[0])

    for label in max_commit_labels:
        Data_frame[label] = [data_points[label]] if label not in Data_frame else Data_frame[label] + [
            data_points[label]]

    for label in owner_commit_labels:
        Data_frame[label] = [data_points[label]] if label not in Data_frame else Data_frame[label] + [
            data_points[label]]

    for label in distinct_commit_labels:
        Data_frame[label] = [data_points[label]] if label not in Data_frame else Data_frame[label] + [
            data_points[label]]

    for label in new_commit_labels:
        Data_frame[label] = [data_points[label]] if label not in Data_frame else Data_frame[label] + [
            data_points[label]]

# ==================================================================================================================== #
# FORK DATA
# ==================================================================================================================== #
    data = get_Git_data(user_pass, forks_url)

    fork_labels = dm.create_labels('forks', months, interval)

    data_points = dm.get_forks_data(dates, data, fork_labels)

    for label in fork_labels:
        Data_frame[label] = [data_points[label]] if label not in Data_frame else Data_frame[label] + [
            data_points[label]]

# ==================================================================================================================== #
# ISSUE DATA
# ==================================================================================================================== #

    data = get_Git_data(user_pass, issues_url)

    o_issues_labels = dm.create_labels('open_issues', months, interval)
    c_issues_labels = dm.create_labels('closed_issues', months, interval)

    data_points = dm.get_issues_data(c_issues_labels, o_issues_labels, dates, data)

    for label in c_issues_labels:
        Data_frame[label] = [data_points[label]] if label not in Data_frame else Data_frame[label] + [
            data_points[label]]

    for label in o_issues_labels:
        Data_frame[label] = [data_points[label]] if label not in Data_frame else Data_frame[label] + [
            data_points[label]]

# ==================================================================================================================== #
# PULL DATA
# ==================================================================================================================== #
    data = get_Git_data(user_pass, pulls_url)

    o_pull_label = dm.create_labels('open_pull', months, interval)
    c_pull_label = dm.create_labels('closed_pull', months, interval)
    m_pull_label = dm.create_labels('merged_pull', months, interval)

    data_points = dm.get_pull_data(o_pull_label, c_pull_label, m_pull_label, dates, data)

    for label in o_pull_label:
        Data_frame[label] = [data_points[label]] if label not in Data_frame else Data_frame[label] + [
            data_points[label]]

    for label in c_pull_label:
        Data_frame[label] = [data_points[label]] if label not in Data_frame else Data_frame[label] + [
            data_points[label]]

    for label in m_pull_label:
        Data_frame[label] = [data_points[label]] if label not in Data_frame else Data_frame[label] + [
            data_points[label]]

# ==================================================================================================================== #
# OWNER DATA
# ==================================================================================================================== #

    data = get_Git_data(user_pass, user_url)


    owner_labels = dm.create_labels('owner_projects', months, interval)

    data_points = dm.get_owner_data(owner_labels, dates, data)

    for label in owner_labels:
        Data_frame[label] = [data_points[label]] if label not in Data_frame else Data_frame[label] + [
            data_points[label]]

    print(Data_frame)
    index += 1

# makes a result.csv for correlation analysis.
df = pd.DataFrame(Data_frame)
df.set_index('Repository', inplace=True)

df.to_csv('/correlationAnalysis/result.csv')