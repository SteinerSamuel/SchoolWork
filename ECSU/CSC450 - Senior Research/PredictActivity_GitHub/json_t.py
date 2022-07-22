"""
This is a module to manipulate GitHub API information
"""
import json
import datetime
import pandas as pd

def get_last_date(data):
    """
    :param data: JSON data to iterate through
    :return: The date of the last commit
    """
    last_date = datetime.datetime.strptime('1996-12-31', '%Y-%m-%d')  # initializes a datetime.datetime object
    for x in data:
        # splits the string from the data to get the date and remove the time
        date_str = x['commit']['author']['date'].split('T')[0]
        new_date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        # tests the date to see if its newer
        if new_date > last_date:
            last_date = new_date
    return last_date


def get_list_of_dates(last_date, length, interval):
    """
    :param last_date: The date of the last commit
    :param length: the amount of months that its looking at
    :param interval: the amount of months each data point looks at
    :return:
    """

    if last_date.day >= 28:
        last_date = last_date.replace(day=28)

    dates = [last_date]  # creates the list of datetime.datetime objects
    m = last_date.month  # sets the initial month
    data_points = length//interval  # number of data points
    year_sub = True if m <= interval else False  # used to change the year when making date labels
    y = 0  # year to subtract
    while data_points > 0:  # loops till we don't need data points
        m = (m - interval) % 12  # gets the date - 3 months 0 =  december
        n = m if m > 0 else 12  # sets n to the month based on m
        y += 1 if year_sub else 0  # if year_sub flag is set it adds 1 to y
        year_sub = True if n <= interval else False  # checks to see if the year_sub flag needs to be active
        dates += [last_date.replace(month=n, year=last_date.year - y)]  # adds the new data point to the list

        data_points -= 1

    return dates


def create_labels(title, length, interval):
    """
    :param title: the title of the feature
    :param length: the amount of months that its looking at
    :param interval: the amount of months each data point looks at
    :return: returns a list of labels
    """
    labels = []
    leng = length
    while leng > 0:
        labels += [title + "_" + str(leng) + "_" + str(leng-interval)]
        leng -= interval
    return labels


def get_commit_data(dates, data, labels):
    """
    :param dates:  list of dates for the function to filter
    :param data:  the json data
    :param labels:  the labels for the data points
    :return: returns a dictionary with the data points
    """
    # makes an emp  ty dictionary for the data
    data_points = {}
    for x in labels:
        data_points[x] = 0

    # gets the length of the date -1 which should be equal to length//interval
    i = 0
    while i < len(dates) - 1:
        for x in data:
            # splits the string from the data to get the date and remove the time
            date_str = x['commit']['author']['date'].split('T')[0]
            compare_date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
            # adds 1 to the data point if it fits within
            if compare_date <= dates[i]:
                if compare_date > dates[i+1]:
                    data_points[labels[i]] += 1
        i += 1

    return data_points


def get_max_days_wo_commit(dates, data, labels):
    data_points = {}
    for x in labels:
        data_points[x] = 0
    i = 0
    while i < len(dates) - 1:
        dates_betweens = []
        for x in data:
            date_str = x['commit']['author']['date'].split('T')[0]
            compare_date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
            if compare_date <= dates[i]:
                if compare_date > dates[i+1]:
                    dates_betweens += [compare_date]

        dates_betweens += [dates[i], dates[i+1]]
        dates_betweens.sort()

        j = 0
        max_days = 0
        while j < len(dates_betweens) - 1:
            days = dates_betweens[j] - dates_betweens[j+1]
            max_days = (abs(days.days)) if abs(days.days) > max_days else max_days
            # max_days = days if days > max_days else max_days
            j += 1

        data_points[labels[i]] = max_days

        i += 1


    return data_points


def get_contrib(dates, data, labels_m, labels_d, labels_o, labels_n, owner):
    data_points = {}
    for label in labels_m:
        data_points[label] = 0

    for label in labels_d:
        data_points[label] = 0

    for label in labels_o:
        data_points[label] = 0

    for label in labels_n:
        data_points[label] =0

    i = len(dates) - 1
    contrib = []
    while i > 0:
        contribs_period = {}
        new_contrib = 0
        # new contributors and making the data
        for x in data:
            date_str = x['commit']['author']['date'].split('T')[0]
            compare_date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
            if compare_date <= dates[i - 1]:
                if compare_date > dates[i]:
                    try:
                        y = x['authror']['login']
                    except TypeError:
                        y = x['commit']['author']['name']
                    except KeyError:
                        y = x['commit']['author']['name']
                    if y not in contrib:
                        contrib += [y]
                        new_contrib += 1
                    if y in contribs_period:
                        contribs_period[y] += 1
                    else:
                        contribs_period[y] = 1
        data_points[labels_d[i-1]] = len(contribs_period)
        try:
            data_points[labels_m[i-1]] = contribs_period[max(contribs_period)]
        except ValueError:
            data_points[labels_m[i-1]] = 0
        try:
            data_points[labels_o[i-1]] = contribs_period[owner]
        except KeyError:
            data_points[labels_o[i-1]] = 0
        data_points[labels_n[i-1]] = new_contrib
        i -= 1
    return data_points


def get_forks_data(dates, data, labels):
    """
    :param dates:  list of dates for the function to filter
    :param data:  the json data
    :param labels:  the labels for the data points
    :return: returns a dictionary with the data points
    """
    data_points = {}
    for x in labels:
        data_points[x] = 0

    # gets the length of the date -1 which should be equal to length//interval
    i = 0
    while i < len(dates) - 1:
        for x in data:
            date_str = x['created_at'].split('T')[0]
            compare_date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
            if compare_date <= dates[i]:
                if compare_date > dates[i + 1]:
                    data_points[labels[i]] += 1
        i += 1

    return data_points


def get_issues_data(label_c, label_o, dates, data):
    """
    :param label_c:
    :param label_o:
    :param dates:
    :param data:
    :return:
    """
    data_points = {}
    for x in label_c:
        data_points[x] = 0


    for x in label_o:
        data_points[x] = 0

    i = 0
    while i < len(dates) - 1:
        for x in data:
            date_str = x['created_at'].split('T')[0]
            compare_date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
            if compare_date <= dates[i]:
                if compare_date > dates[i + 1]:
                    data_points[label_o[i]] += 1
        i += 1

    i = 0
    while i < len(dates) - 1:
        for x in data:
            if x['closed_at'] == None:
                date_str = "1001-01-01"
            else:
                date_str = x['closed_at'].split('T')[0]
            compare_date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
            if compare_date <= dates[i]:
                if compare_date > dates[i + 1]:
                    data_points[label_c[i]] += 1
        i += 1

    return data_points


def get_pull_data(label_o, label_c, label_m, dates, data):
    data_points = {}

    for x in label_c:
        data_points[x] = 0

    for x in label_o:
        data_points[x] = 0

    for x in label_m:
        data_points[x] = 0

    i = 0
    while i < len(dates) - 1:
        for x in data:
            date_str = x['created_at'].split('T')[0]
            compare_date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
            if compare_date <= dates[i]:
                if compare_date > dates[i + 1]:
                    data_points[label_o[i]] += 1
        i += 1

    i = 0
    while i < len(dates) - 1:
        for x in data:
            if x['closed_at'] == None:
                date_str = "1001-01-01"
            else:
                date_str = x['closed_at'].split('T')[0]
            compare_date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
            if compare_date <= dates[i]:
                if compare_date > dates[i + 1]:
                    data_points[label_c[i]] += 1
        i += 1

    i = 0
    while i < len(dates) - 1:
        for x in data:
            if x['merged_at'] == None:
                date_str = "1001-01-01"
            else:
                date_str = x['merged_at'].split('T')[0]
            compare_date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
            if compare_date <= dates[i]:
                if compare_date > dates[i + 1]:
                    data_points[label_m[i]] += 1
        i += 1

    return data_points


def get_owner_data(labels, dates, data):
    data_points = {}
    for x in labels:
        data_points[x] = 0

    # gets the length of the date -1 which should be equal to length//interval
    i = 0
    while i < len(dates) - 1:
        for x in data:
            # splits the string from the data to get the date and remove the time
            date_str = x['created_at'].split('T')[0]
            compare_date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
            # adds 1 to the data point if it fits within
            if compare_date <= dates[i]:
                if compare_date > dates[i + 1]:
                    data_points[labels[i]] += 1
        i += 1

    return data_points

# this should only be run as main to test with the provided test data
if __name__ == '__main__':
    json_df = {'repository_name': []}
    json_df['repository_name'] += ['itsabot/itsabot']

    length = 24
    interval = 3
    # Commit DATA
    with open("test_files/commit_test.json") as json_test:
        data = json.load(json_test)

    last_date = get_last_date(data)

    dates = get_list_of_dates(last_date, length, interval)

    commit_labels = create_labels('commit', length, interval)

    data_points = get_commit_data(dates, data, commit_labels)
    for label in commit_labels:
        json_df[label] = [data_points[label]]

    commit_days_labels = create_labels('max_days_since', length, interval)

    data_points = get_max_days_wo_commit(dates,data, commit_days_labels)

    for label in commit_days_labels:
        json_df[label] = [data_points[label]]


    max_commit_labels = create_labels('max_commit', length, interval)
    owner_commit_labels = create_labels('owner_commit', length, interval)
    distinct_commit_labels = create_labels('dis_commit', length, interval)
    new_commit_labels = create_labels('new_commit', length, interval)

    data_points = get_contrib(dates, data, max_commit_labels, distinct_commit_labels, owner_commit_labels,
                              new_commit_labels, 'itsabot')

    for label in max_commit_labels:
        json_df[label] = [data_points[label]]

    for label in owner_commit_labels:
        json_df[label] = [data_points[label]]

    for label in distinct_commit_labels:
        json_df[label] = [data_points[label]]

    for label in new_commit_labels:
        json_df[label] = [data_points[label]]


    # Forks DATA
    with open("test_files/forks_test.json") as json_test:
        data = json.load(json_test)

    fork_labels = create_labels('forks', length, interval)

    data_points = get_forks_data(dates, data, fork_labels)
    for label in fork_labels:
        json_df[label] = [data_points[label]]

    # ISSUES DATA
    with open("test_files/issues_test.json") as json_test:
        data = json.load(json_test)

    o_issues_labels = create_labels('open_issues', length, interval)
    c_issues_labels = create_labels('closed_issues', length, interval)

    data_points = get_issues_data(c_issues_labels, o_issues_labels, dates, data)

    for label in c_issues_labels:
        json_df[label] = [data_points[label]]

    for label in o_issues_labels:
        json_df[label] = [data_points[label]]

    # PULLS DATA
    with open("test_files/pulls_Test.json") as json_test:
        data = json.load(json_test)

    o_pull_label = create_labels('open_pull', length, interval)
    c_pull_label = create_labels('closed_pull', length, interval)
    m_pull_label = create_labels('merged_pull', length, interval)

    data_points = get_pull_data(o_pull_label, c_pull_label, m_pull_label, dates, data)

    for label in o_pull_label:
        json_df[label] = [data_points[label]]

    for label in c_pull_label:
        json_df[label] = [data_points[label]]

    for label in m_pull_label:
        json_df[label] = [data_points[label]]

    # owner DATA
    with open('test_files/test.json') as json_test:
        data = json.load(json_test)

    owner_labels = create_labels('owner_projects', length, interval)

    data_points = get_owner_data(owner_labels, dates, data)

    for label in owner_labels:
        json_df[label] = [data_points[label]]

    print(json_df)

    df = pd.DataFrame(json_df)
    df.set_index('repository_name', inplace=True)

    df.to_csv('test.csv')
