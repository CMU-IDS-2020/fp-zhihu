from altair import datum
from collections import Counter
from datetime import datetime
from google.cloud import bigquery
from global_var import table_name, user_id_str, date_str
from global_var import MAX_WIDTH, PREFIX, recommend, safe_config
from multiprocessing import Pool
import altair as alt
import base64
import numpy as np
import os
import pandas as pd
import streamlit as st
import time


def serialization(id):
    if not isinstance(id, list):
        id = [id]
    if isinstance(id[0], str):
        id = ["'" + i + "'" for i in id]
    else:
        id = [str(i) for i in id]
    return '(' + ','.join(id) + ')'


@st.cache
def get_query(query):
    print(time.asctime(), query)
    return bigquery.Client().query(
        query, job_config=safe_config).to_dataframe()


@st.cache
def qid2title(qid):
    qid = sorted(list(set(qid)))
    query1 = f"""
        SELECT id, parent_id from `{PREFIX}.posts_answers`
        WHERE id in {serialization(qid)}"""
    query2 = f"""
        SELECT id, title from `{PREFIX}.posts_questions`
        WHERE id in {serialization(qid)}"""
    df1, df2 = get_query(query1), get_query(query2)
    # with Pool(2) as p:
    #     df1, df2 = p.map(get_query, [query1, query2])
    aid2qid = {a: p for a, p in zip(df1['id'], df1['parent_id'])}
    qid2title = {i: t for i, t in zip(df2['id'], df2['title'])}
    return {q: qid2title.get(aid2qid.get(q, q), "this question") for q in qid}


@st.cache
def get_user_info(user_id):
    if not isinstance(user_id, list):
        user_id = [user_id]
    if len(user_id) == 0:
        return None
    return get_query(f'SELECT * FROM `{PREFIX}.users` WHERE id in {serialization(user_id)}')


@st.cache
def get_user_timeline(
    user_id, start=datetime(2008, 8, 1), end=datetime(2020, 9, 10)
):
    if not isinstance(user_id, list):
        user_id = [user_id]
    if len(user_id) == 0:
        return {}
    start, end = start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')
    query_list = []
    for t, u, d in zip(table_name, user_id_str, date_str):
        query = f'SELECT * FROM `{PREFIX}.{t}` WHERE {u} IN {serialization(user_id)}'
        if d != '/':
            query += f" AND {d} BETWEEN '{start}' AND '{end}'"
        query_list.append(query)
    user_Q = get_query(query_list[-1])
    if len(user_Q) == 0:
        return {uid: {'users': []} for uid in user_id}
    raw = [get_query(q) for q in query_list[:-1]] + [user_Q]
    # with Pool(len(query_list)) as p:
    #     raw = p.map(get_query, query_list)
    result = {}
    for uid in user_id:
        result[uid] = {}
        for t, u, r in zip(table_name, user_id_str, raw):
            result[uid][t] = r.loc[r[u] == uid]
    return result


def getweek(t: datetime) -> int:
    w = t.weekofyear
    if t.day_name() == 'Sunday':
        w += 1
    return w


def get_single_user_timeline(data: dict, ph=st) -> None:
    """
    :param data: the piece of output of get_user_timeline, {table: df}
    :param ph: placeholder, whether to write the result
    """
    if len(data['users']) == 0:
        st.write("No such user :(")
        return
    result = []
    for d, v in zip(date_str, data.values()):
        if d != '/':
            result += v[d].to_list()
    if len(result) == 0:
        st.write("This user has literary no record :(")
        return
    uid = data["users"]["id"][0]
    name = data["users"]["display_name"][0]
    st.write(f'User\'s Stack Overflow Page: [{name}](https://stackoverflow.com/users/{uid})')

    # draw bubble
    raw = []
    # fill valid data
    for t in result:
        raw.append((t, t.year, getweek(t), 1))
    # fill null data
    t, delta, end = pd.Timestamp(2008, 1, 1), pd.Timedelta(
        days=1), pd.Timestamp(2020, 12, 31)
    while t < end:
        raw.append((t, t.year, getweek(t), 0))
        t += delta
    slider = alt.binding_range(min=2008, max=2020, step=1, name='Year: ')
    selector = alt.selection_single(name="SelectorName", fields=['year'],
                                    bind=slider, init={'year': 2020})
    df = pd.DataFrame(raw, columns=['time', 'year', 'week', 'Contribution'])
    scale = alt.Scale(
        range=["#F0F0F0", 'white', 'green'],
        domain=[0, 1, 10]
    )
    ph.write(alt.Chart(df).mark_rect().encode(
        x=alt.X('week:O', axis=alt.Axis(title='Week')),
        y=alt.Y('day(time):O', axis=alt.Axis(title='Day')),
        color=alt.Color('sum(Contribution):Q',
                        scale=scale),
        tooltip=[
            alt.Tooltip('yearmonthdate(time)', title='Date'),
            alt.Tooltip('sum(Contribution)', title='Contribution'),
        ],
    ).add_selection(selector).transform_filter(selector).properties(
        title=f'{name}\'s Contributions', width=MAX_WIDTH, height=150
    ).configure_scale(bandPaddingInner=0.2))

    def transform_dt(dt, c=0):
        return [dt, str(dt.isoweekday()) + '-' +
                dt.strftime("%A"), dt.hour, dt.month, dt.year, c]
    # draw other figures
    raw = [transform_dt(i, 1) for i in result]
    # append null data
    for i in range(24):
        dt = datetime(2020, 1, 1, i, 0)
        raw.append(transform_dt(dt))
    for i in range(7):
        dt = datetime(2020, 1, i + 1)
        raw.append(transform_dt(dt))
    for i in range(12):
        dt = datetime(2020, i + 1, 1)
        raw.append(transform_dt(dt))
    for i in range(2008, 2020):
        dt = datetime(i, 1, 1)
        raw.append(transform_dt(dt))

    df = pd.DataFrame(
        raw, columns=['time', 'Date', 'Hour', 'Month', 'Year', 'Contribution'])

    def get_max_index(key):
        dff = df.groupby([key], as_index=False)['Contribution'].sum()
        max_index = dff['Contribution'].argmax()
        max_index = dff.loc[max_index, key]
        if key == 'Date':
            return f"He/She works most on <b>{max_index[2:]}</b>."
        elif key == 'Hour':
            if max_index >= 0 and max_index <= 6:
                hour_name = 'early in the morning'
            elif max_index > 6 and max_index <= 12:
                hour_name = 'in the morning'
            elif max_index > 12 and max_index <= 18:
                hour_name = 'in the afternoon'
            elif max_index > 18 and max_index <= 22:
                hour_name = 'at night'
            elif max_index > 22 and max_index <= 24:
                hour_name = 'at midnight'
            return f"He/She would like to work <b>{hour_name}</b>."
        elif key == 'Month':
            month_number = str(max_index)
            datetime_object = datetime.strptime(month_number, "%m")
            month_name = datetime_object.strftime("%B")
            return f"He/She works most actively in <b>{month_name}</b>."
        elif key == 'Year':
            return f"He/She contributed most in <b>{max_index}</b>."

    st.subheader('User Habits 🧐')
    options = st.multiselect(
        'Find contribution over', ['Hour', 'Date', 'Month', 'Year'], ['Hour', 'Date'])
    chart = {}
    for key in ['Date', 'Hour', 'Month', 'Year']:
        chart[key] = alt.Chart(df).mark_bar().encode(
            x=alt.X(key + ':N'),
            y=alt.Y('sum(Contribution)', axis=alt.Axis(title='Contribution')),
            tooltip=[
                key,
                alt.Tooltip('sum(Contribution)', title='Contribution'),
            ]
        ).properties(
            title=f'{name}\'s Contributions over {key}', height=300, width=500)
    for o in options:
        ph.write(chart[o])
        ph.write(get_max_index(o), unsafe_allow_html=True)
        ph.write('----')


def convert(title: str) -> str:
    return title.replace('<', '').replace('>', '')


def get_multi_user_timeline(data: dict, baseuser) -> None:
    """
    :param data: the full output of get_user_timeline, {uid: {table: df}}
    """
    uid2name = {u: data[u]['users']["display_name"].tolist()[0] for u in data}
    baseuser = f'<a href="https://stackoverflow.com/users/{baseuser["id"].tolist()[0]}/" target="_blank">{baseuser["display_name"].tolist()[0]}</a>'
    table_date = {t: d for t, d in zip(table_name, date_str)}
    qid = []
    qid2t = {}
    raw = []
    for uid, user_data in data.items():
        for tname, table in user_data.items():
            if tname == 'users':
                continue
            for _, i in table.iterrows():
                raw_single = [i[table_date[tname]], tname, uid]
                if tname == 'badges':
                    raw_single += [[i['name']]]
                elif tname == 'comments':
                    raw_single += [[i['post_id']]]
                    qid.append(i['post_id'])
                elif tname == 'posts_answers':
                    raw_single += [[i['parent_id']]]
                    qid.append(i['parent_id'])
                elif tname == 'posts_questions':
                    raw_single += [[i['id']]]
                    qid2t[i['id']] = i['title']
                raw.append(raw_single)
    if len(qid) > 0:
        qid2t.update(qid2title(qid))
    raw = sorted(raw)[::-1]
    stack = []
    for info in raw:
        if len(stack) == 0:
            stack.append(info)
            continue
        tm1, tname1, uid1, info1 = info
        tm0, tname0, uid0, info0 = stack[-1]
        delta = pd.Timedelta(days=2)
        if tm1 - tm0 >= delta or tname1 != tname0 or uid1 != uid0:
            stack.append(info)
            continue
        # merge
        stack[-1] = [tm1, tname0, uid0, info0 + info1]
    # build html
    template = """
<div class="card border-light">
  <div class="card-body">
    <h5 class="card-title">
      <i class="fa ICON" aria-hidden="true"></i> USERNAME
    </h5>
    <p class="card-text">TEXT</p>
    <p class="card-text text-right"><small class="text-muted">TIME</small></p>
  </div>
</div>
<div class="container py-1"></div>"""
    content = ''
    for info in stack:
        tm, tname, uid, info = info
        username = f'<a href="https://stackoverflow.com/users/{uid}/" target="_blank">{uid2name[uid]}</a>'
        if tname == 'badges':
            icon = 'fa-check-square-o'
            info = 'Your friend got ' + ', '.join([f'{v}x {k} badge{"s" if v > 1 else ""}' for k, v in Counter(info).items()]) + ', congratulations!'
        elif tname == 'comments':
            icon = 'fa-commenting-o'
            info = 'Your friend posted ' + ', '.join([f'{v} comment{"s" if v > 1 else ""} in <a href="https://stackoverflow.com/questions/{k}" target="_blank">{convert(qid2t.get(k, "this question"))}</a>' for k, v in Counter(info).items()]) + '.'
        elif tname == 'posts_answers':
            icon = 'fa-align-left'
            info = 'Your friend posted ' + ', '.join([f'{v} answer{"s" if v > 1 else ""} in <a href="https://stackoverflow.com/questions/{k}" target="_blank">{convert(qid2t.get(k, "this question"))}</a>' for k, v in Counter(info).items()]) + '.'
        elif tname == 'posts_questions':
            icon = 'fa-question-circle-o'
            info = f'Your friend posted {len(info)} question{"s" if len(info) > 1 else ""}: ' + ', '.join([f'<a href="https://stackoverflow.com/questions/{k}" target="_blank">{convert(qid2t[k])}</a>' for k in info]) + '.'
        content += template.replace('ICON', icon).replace('USERNAME', username).replace(
            'TIME', tm.strftime("%Y/%m/%d")).replace('TEXT', info)
    st.components.v1.html("""
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css" integrity="sha384-wvfXpqpZZVQGK6TAh5PVlGOfQNHSoD2xbE+QkPxCAFlNEevoEH3Sl0sibVcOQVnN" crossorigin="anonymous">
<script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
<style>
.card{border-radius: 4px;background: #fff;box-shadow: 0 6px 10px rgba(0,0,0,.08), 0 0 6px rgba(0,0,0,.05);transition: .3s transform cubic-bezier(.155,1.105,.295,1.12),.3s box-shadow,.3s -webkit-transform cubic-bezier(.155,1.105,.295,1.12);padding: 14px 80px 18px 36px;cursor: pointer;}
.card:hover{transform: scale(1.05);box-shadow: 0 10px 20px rgba(0,0,0,.12), 0 4px 8px rgba(0,0,0,.06);}
</style>
<div class="text-center py-5"><button type="button" class="btn btn-primary" data-toggle="modal" data-target=".bd-example-modal-lg">Launch App!</button></div>
<div class="modal fade bd-example-modal-lg" tabindex="-1" role="dialog" aria-labelledby="myLargeModalLabel" aria-hidden="true">
  <div class="modal-dialog modal-lg">
    <div class="modal-content">
      <div class="mx-5 py-5">
      <div class="card border-light">
        <div class="card-body">
          <h3 class="card-text">Social Overflow (friends of BASEUSER)</h3>
        </div>
      </div>
      <div class="container py-1"></div>
        CONTENT
        <div class="text-center py-1">
          <small class="text-muted">No more result</small>
        </div>
      </div>
    </div>
  </div>
</div>
    """.replace('CONTENT', content).replace("BASEUSER", baseuser), height=900, scrolling=True)


@st.cache
def get_answer_time_for_each_tag(tags):
    tags = serialization(tags)
    # change query date range
    questions_query = f"""
    WITH question_answers_join AS (
      SELECT *
        , GREATEST(1, TIMESTAMP_DIFF(answers.first, creation_date, minute)) minutes_2_answer
      FROM (
        SELECT id, creation_date, title
          , (SELECT AS STRUCT MIN(creation_date) first, COUNT(*) c
             FROM `{PREFIX}.posts_answers` 
             WHERE a.id=parent_id
          ) answers
          , SPLIT(tags, '|') tags
        FROM `{PREFIX}.posts_questions` a
        WHERE EXTRACT(year FROM creation_date) > 2008
      )
    )
    SELECT COUNT(*) questions, tag
      , ROUND(EXP(AVG(LOG(minutes_2_answer))), 2) avg_minutes
      , COUNT(minutes_2_answer)/COUNT(*) chance_of_answer
    FROM question_answers_join, UNNEST(tags) tag
    WHERE tag IN {tags}
    GROUP BY tag
    ORDER BY avg_minutes
    """
    questions_query_job = bigquery.Client().query(
        questions_query, job_config=safe_config)

    # API request - run the query, and return a pandas DataFrame
    questions_results = questions_query_job.to_dataframe()
    # return questions_results.set_index('tag').T.to_dict('dict')
    return questions_results


@st.cache
def get_answer_time_for_all_tags(tags):
    tags = ["tags LIKE '%" + t + "%'"for t in tags]
    tags = " AND ".join(tags)
    # change query date range
    questions_query = f"""
    WITH question_answers_join AS (
      SELECT *
        , GREATEST(1, TIMESTAMP_DIFF(answers.first, creation_date, minute)) minutes_2_answer
      FROM (
        SELECT id, creation_date, title
          , (SELECT AS STRUCT MIN(creation_date) first, COUNT(*) c
             FROM `{PREFIX}.posts_answers`
             WHERE a.id=parent_id
          ) answers
          , tags
        FROM `{PREFIX}.posts_questions` a
        WHERE EXTRACT(year FROM creation_date) > 2008
      )
    )
    SELECT COUNT(*) questions, ROUND(EXP(AVG(LOG(minutes_2_answer))), 2) avg_minutes
      , COALESCE(COUNT(minutes_2_answer)/NULLIF(COUNT(*) , 0), 0) chance_of_answer
    FROM question_answers_join
    WHERE {tags}
    """
    questions_query_job = bigquery.Client().query(
        questions_query, job_config=safe_config)

    # API request - run the query, and return a pandas DataFrame
    questions_results = questions_query_job.to_dataframe()
    res = {
        'num_of_question': questions_results.loc[0, 'questions'],
        'avg_minutes': questions_results.loc[0, 'avg_minutes'],
        'chance_of_answer': questions_results.loc[0, 'chance_of_answer']
    }
    return res


def show_estimated_times(tags):
    if st.checkbox('Show estimated answer time and answer rate for each tag'):
        df = get_answer_time_for_each_tag(tags)
        st.write(alt.Chart(df).mark_circle().encode(
            x=alt.X('avg_minutes:Q', axis=alt.Axis(title='Time (minutes)')),
            y=alt.Y('chance_of_answer:Q', axis=alt.Axis(title='Probability')),
            size=alt.Size('questions:Q', legend=None),
            color=alt.Color('tag:N'),
            tooltip=[
                alt.Tooltip('tag'),
                alt.Tooltip('avg_minutes'),
                alt.Tooltip('chance_of_answer'),
                alt.Tooltip('questions'),
            ],
        ).properties(title=f'Estimated answer time and answer rate for each tag'))

    res = get_answer_time_for_all_tags(tags)
    if res['avg_minutes'] != np.nan:
        st.write(f"<font color=black>The estimated answer time for these tags combined is <b>{'{0:.0f}'.format(res['avg_minutes'])}</b> minutes.</font>", unsafe_allow_html=True)
        st.write(f"The estimated answer rate for these tags combined is <b>{'{0:.2f}%'.format(res['chance_of_answer']*100)}</b>.", unsafe_allow_html=True)
    else:
        st.write(f"Sorry, no estimation because you are the first one to ask these types of question. ")

def SO_current_situation():

    st.write("<span style='font-size:30px;'>Stack Overflow</span> is the largest online community for programmers to learn, share their knowledge, and advance their careers.", unsafe_allow_html=True)
    st.write("Currently, it has over 10,000,000 registered users in the community who can: ")
    st.write("""
            ✅ Ask and Answer Questions \n
            ✅ Vote Questions and Answers Up or Down  \n
            ✅ Edit Other People's Posts \n
            ❓ What's more.... 
            """)

def narrative():
    st.write('# Stack Overflow Helper')
    st.markdown('''
        > GitHub project page: https://github.com/CMU-IDS-2020/fp-zhihu

        > Dataset credit: [Kaggle](https://www.kaggle.com/stackoverflow/stackoverflow)
    ''')
    st.markdown('---')
 
    SO_current_situation()

    st.write('## What\'s the problem right now?')

    qpy = get_query("""
        SELECT EXTRACT(YEAR FROM creation_date) AS year, COUNT(*) AS cnt
        FROM `bigquery-public-data.stackoverflow.posts_questions`
        GROUP BY year
        ORDER BY year
    """)
    aqpy = get_query("""
        SELECT EXTRACT(YEAR FROM creation_date) AS year, COUNT(*) AS cnt
        FROM `bigquery-public-data.stackoverflow.posts_questions`
        WHERE answer_count > 0
        GROUP BY year
        ORDER BY year
    """)
    result = 1 - aqpy['cnt'].sum() / qpy['cnt'].sum()
    st.markdown(f'The biggest **_problem_** SO users face is that **_{result * 100:.2f}%_** of questions are not answered, and the trend is still going up!')
    df = pd.DataFrame([
        (y, n, n / n, 'Total Questions') for y, n, na in zip(qpy['year'], qpy['cnt'], aqpy['cnt'])
    ] + [
        (y, na, na / n, 'Questions have answers') for y, n, na in zip(aqpy['year'], qpy['cnt'], aqpy['cnt'])
    ], columns=['Year', 'Count', 'Percentage', 'Type'])
    st.write(alt.Chart(df).mark_bar().encode(
        x=alt.X('Year:N'),
        y=alt.Y('max(Count)', stack=None),
        color='Type',
        tooltip=['Year', 'Count', 'Type', 'Percentage']
    ).properties(title='Question Count', width=MAX_WIDTH))

    st.header("Two Observations")
    st.markdown(
        '1. Different **_tags_** tend to have different answer rates and different average minutes to get an answer.')

    all_tags = get_query("""
        SELECT tag_name
        FROM `bigquery-public-data.stackoverflow.tags`
        WHERE count >= 1000
        ORDER BY RAND()
        LIMIT 15
    """)['tag_name'].tolist()

    answer_time = get_answer_time_for_each_tag(all_tags)

    base = alt.Chart(answer_time).encode(
        x=alt.X('tag', sort=None, title='Tag'))
    left = base.mark_bar().encode(
        y=alt.Y('avg_minutes', title='Avg minutes to answer'),
    )
    right = base.mark_line(color='red').encode(
        y=alt.Y('chance_of_answer', title='Chance of answer',
                scale=alt.Scale(zero=False)),
    )
    st.write(alt.layer(left, right).encode(
        tooltip=['tag', 'avg_minutes', 'chance_of_answer']
    ).resolve_scale(y='independent').properties(width=MAX_WIDTH))

    st.markdown(
        '2. A user tends to answer questions with **_some particular tags_**.')
    uid = int(st.text_input('Input user id', '1694'))
    tags_one_user = get_query(f"""
        SELECT questions.tags
        FROM `bigquery-public-data.stackoverflow.posts_answers` answers INNER JOIN
        `bigquery-public-data.stackoverflow.posts_questions` questions ON answers.parent_id = questions.id
        WHERE answers.owner_user_id = {uid}
        -- GROUP BY tags.tag_name
    """)['tags'].tolist()
    tag_cnt = dict()
    for tags in tags_one_user:
        for tag in tags.split('|'):
            if tag in tag_cnt.keys():
                tag_cnt[tag] += 1
            else:
                tag_cnt[tag] = 1

    tag_cnt = sorted(tag_cnt.items(), key=lambda item: item[1])
    x, y = [], []
    for idx in range(len(tag_cnt)):
        x.append(idx // 26)
        y.append(idx % 26)
    tag_cnt = pd.DataFrame.from_records(tag_cnt, columns=['tag', 'count'])
    tag_cnt['x'] = x
    tag_cnt['y'] = y
    tag_cnt_plot = alt.Chart(tag_cnt).mark_circle(size=200).encode(
        x=alt.X('x', axis=alt.Axis(labels=False, title=f'Answer Counts For Tags For User {uid}')),
        y=alt.Y('y', axis=alt.Axis(labels=False, title="")),
        size='count',
        tooltip=['tag', 'count']
    ).properties(width=MAX_WIDTH, height=350)
    tag_cnt_plot

    st.markdown("""
        In order to **_increase answer rate_** and **_reduce answer time_**, the helper recommends 
        users to answer particular questions based on the **_tags_**. 
    """)


def tag_user_recommendation():
    st.header('Recommendations based on question tags')
    question = st.text_input('Input question (Optional)',
                             "How to convert pandas dataframe to numpy array?")
    tags = [t.strip() for t in st.text_input(
            'Input question tags, "," seperated', "python, pandas, numpy").split(',')]
    show_estimated_times(tags)
    tag_id = recommend.get_tag_id_by_name(tags)

    st.header('Recommended users for your question')
    user_num = st.slider(
        'Select the top k users recommended for you:', 0, 20, 5)
    user_id = recommend.get_recommendation_by_tag_id(tag_id, k=user_num)
    user_df = get_user_info(user_id.tolist())
    if st.checkbox('Show raw data for recommended users'):
        st.write(user_df)
    user_detail = st.checkbox('Show details for each user')
    for i, uid in enumerate(user_id):
        row = user_df.loc[user_df['id'] == uid]
        username = row['display_name'].values[0]
        intro = row['about_me'].values[0]
        st.subheader(f'Top {i+1}: [{username}](https://stackoverflow.com/users/{uid})')
        if intro and user_detail:
            st.write(intro, unsafe_allow_html=True)
        elif user_detail:
            st.write('No introduction provided')


def single_user():
    st.markdown("# Personal Profile Page")
    uid = int(st.text_input('Input user id', '16241'))
    user_data = get_user_timeline([uid])
    get_single_user_timeline(user_data[uid])


def multi_user():
    st.markdown("# Social Overflow")
    col1, col2 = st.beta_columns([1, 3])
    with col1:
        base = int(st.text_input('Input base user id', '3122'))
    with col2:
        friend = list(map(int, st.text_input(
            'Input friend users id, "," seperated', "2686, 2795, 4855").split(',')))

    st.header('Recommended users for following')
    recommend.reset_followings(base, friend)
    user_num = st.slider(
        'Select the top k users recommended for you:', 0, 20, 5)
    user_id = recommend.recommend_users_by_history(base, k=user_num)
    user_df = get_user_info(user_id.tolist())
    if st.checkbox('Show raw data for recommended users'):
        st.write(user_df)
    user_detail = st.checkbox('Show details for each user')
    for i, uid in enumerate(user_id):
        row = user_df.loc[user_df['id'] == uid]
        username = row['display_name'].values[0]
        intro = row['about_me'].values[0]
        st.subheader(f'Top {i+1}: [{username}](https://stackoverflow.com/users/{uid}), (user id {uid})')
        if intro and user_detail:
            st.write(intro, unsafe_allow_html=True)
        elif user_detail:
            st.write('No introduction provided')

    st.markdown('---')
    timestamp = st.slider(
        'Select date range', datetime(2008, 8, 1), datetime(2020, 9, 10),
        value=(datetime(2019, 9, 10), datetime(2020, 9, 10)),
        format="MM/DD/YY")
    user_data = get_user_timeline([base] + friend, *timestamp)

    get_multi_user_timeline(
        {i: user_data[i] for i in friend}, user_data[base]['users'])


def user_user_recommendation():
    st.sidebar.markdown('---')
    function_mapping = {
        'Personal Profile Page': lambda: single_user(),
        'User Recommend and Timeline': lambda: multi_user(),
    }
    option = st.sidebar.selectbox("Page", list(function_mapping.keys()))
    function_mapping[option]()


def main():
    function_mapping = {
        'Project Description and Motivation': lambda: narrative(),
        'Tag-User Recommendation': lambda: tag_user_recommendation(),
        'User-User Recommendation': lambda: user_user_recommendation(),
    }
    option = st.sidebar.selectbox("Navigation", list(function_mapping.keys()))
    function_mapping[option]()


if __name__ == '__main__':
    main()
