import os, json
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import plotly.express as px
from scipy.sparse import csr_matrix

st.set_page_config(page_title='Late Delivery Risk',
                   layout='wide')

@st.cache_data
def load_orders():
    name = 'scored_orders.csv.gz'
    if not os.path.exists(name):
        name = 'scored_orders.csv'
    return pd.read_csv(name)

@st.cache_resource
def load_model():
    booster = xgb.Booster()
    booster.load_model('xgb_model.json')
    prep = json.load(open('preprocess.json'))
    return booster, prep

df = load_orders()
booster, prep = load_model()
metrics = json.load(open('metrics.json'))
drivers = pd.read_csv('risk_drivers.csv')
PROB = 'Late Delivery Probability'

def to_matrix(row):
    mean = np.array(prep['num_mean'])
    scale = np.array(prep['num_scale'])
    nums = row[prep['num_cols']].astype(float).values
    parts = [(nums - mean) / scale]
    for c in prep['cat_cols']:
        values = prep['cat_values'][c]
        hot = [1.0 if v == row[c].iloc[0] else 0.0
               for v in values]
        parts.append(np.array([hot]))
    return csr_matrix(np.hstack(parts))

def pick(label, col):
    options = sorted(df[col].unique())
    return st.sidebar.multiselect(label, options,
                                  default=options)

st.sidebar.title('Filters')
modes = pick('Shipping mode', 'Shipping Mode')
markets = pick('Market', 'Market')
regions = pick('Order region', 'Order Region')
segments = pick('Customer segment', 'Customer Segment')
threshold = st.sidebar.slider(
    'Risk threshold (High risk if above)',
    0.10, 0.99, 0.70, 0.01)

f = df[df['Shipping Mode'].isin(modes)
       & df['Market'].isin(markets)
       & df['Order Region'].isin(regions)
       & df['Customer Segment'].isin(segments)].copy()
if f.empty:
    st.warning('No orders match these filters.')
    st.stop()
f['High'] = f[PROB] >= threshold

page = st.sidebar.radio('Dashboard module', [
    'Delay Risk Overview',
    'Order-Level Risk Prediction',
    'Region & Mode Risk Analysis',
    'Operations Action Panel'])
st.title('Late Delivery Risk - APL Logistics')

if page == 'Delay Risk Overview':
    a, b, c, d = st.columns(4)
    a.metric('Orders in view', f'{len(f):,}')
    b.metric('High-risk orders', f"{int(f['High'].sum()):,}")
    c.metric('Avg late probability', f'{f[PROB].mean():.1%}')
    auc = metrics['XGBoost']['ROC-AUC']
    d.metric('Model ROC-AUC', f'{auc:.3f}')

    order = ['Low Risk', 'Medium Risk', 'High Risk']
    counts = f['Risk Category'].value_counts()
    counts = counts.reindex(order).reset_index()
    counts.columns = ['Risk Category', 'Orders']
    colors = {'Low Risk': 'green',
              'Medium Risk': 'orange',
              'High Risk': 'red'}
    fig = px.bar(counts, x='Risk Category', y='Orders',
                 color='Risk Category',
                 color_discrete_map=colors,
                 title='Overall risk distribution')
    st.plotly_chart(fig, width='stretch')

    fig2 = px.bar(drivers.head(8).iloc[::-1],
                  x='Importance', y='Feature',
                  orientation='h',
                  title='Key Risk Drivers')
    st.plotly_chart(fig2, width='stretch')

elif page == 'Order-Level Risk Prediction':
    st.subheader('Score one order before it ships')
    c1, c2, c3 = st.columns(3)
    days = {'Standard Class': 4, 'Second Class': 2,
            'First Class': 1, 'Same Day': 0}
    mode = c1.selectbox('Shipping mode', list(days))
    market = c1.selectbox('Market',
                          sorted(df['Market'].unique()))
    region = c1.selectbox('Order region', sorted(
        df[df['Market'] == market]['Order Region'].unique()))
    seg = c2.selectbox('Customer segment',
                       sorted(df['Customer Segment'].unique()))
    ptype = c2.selectbox('Payment type',
                         sorted(df['Type'].unique()))
    dept = c2.selectbox('Department',
                        sorted(df['Department Name'].unique()))
    qty = c3.number_input('Quantity', 1, 5, 2)
    price = c3.number_input('Product price', 5.0, 2000.0, 100.0)
    rate = c3.slider('Discount rate', 0.0, 0.25, 0.05)
    profit = c3.slider('Profit ratio', -1.0, 1.0, 0.2)

    sched = days[mode]
    row = pd.DataFrame([{
        'Shipping Mode': mode, 'Market': market,
        'Order Region': region, 'Customer Segment': seg,
        'Type': ptype, 'Department Name': dept,
        'sched_days': sched,
        'shipping_pressure': qty / (sched + 1),
        'is_express': int(mode in ['First Class', 'Same Day']),
        'order_complexity': qty * price * (1 + rate) / 100,
        'Order Item Discount Rate': rate,
        'Order Item Profit Ratio': profit,
        'Order Item Quantity': qty,
        'Order Item Product Price': price,
        'Sales': qty * price,
        'Order Item Discount': qty * price * rate,
        'region_congestion': prep['reg_rate'].get(
            region, prep['global_rate'])}])

    M = xgb.DMatrix(to_matrix(row))
    p = float(booster.predict(M)[0])
    if p < 0.4:
        cat = 'Low Risk'
    elif p <= 0.7:
        cat = 'Medium Risk'
    else:
        cat = 'High Risk'
    k1, k2 = st.columns(2)
    k1.metric('Late Delivery Probability', f'{p:.1%}')
    k2.metric('Risk Category', cat)

    contrib = booster.predict(M, pred_contribs=True)[0]
    groups = {}
    pos = 0
    for c in prep['num_cols']:
        groups[c] = float(contrib[pos])
        pos += 1
    for c in prep['cat_cols']:
        n = len(prep['cat_values'][c])
        groups[c] = float(contrib[pos:pos + n].sum())
        pos += n
    why = pd.Series(groups).sort_values(key=abs,
                                        ascending=False)
    why = why.head(6).reset_index()
    why.columns = ['Factor', 'Impact']
    why['Effect'] = why['Impact'].apply(
        lambda v: 'raises risk' if v > 0 else 'lowers risk')
    fig = px.bar(why, x='Impact', y='Factor',
                 color='Effect', orientation='h',
                 title='Key contributing factors',
                 color_discrete_map={'raises risk': 'red',
                                     'lowers risk': 'green'})
    st.plotly_chart(fig, width='stretch')
    st.write('Biggest reason:', why.loc[0, 'Factor'],
             why.loc[0, 'Effect'])

elif page == 'Region & Mode Risk Analysis':
    heat = f.pivot_table(index='Order Region',
                         columns='Shipping Mode',
                         values=PROB, aggfunc='mean')
    fig = px.imshow(heat, aspect='auto',
                    color_continuous_scale='RdYlGn_r',
                    title='Risk heatmap: region x shipping mode')
    st.plotly_chart(fig, width='stretch')

    g = f.groupby('Shipping Mode')
    g = g[[PROB, 'Late_delivery_risk']].mean().reset_index()
    fig2 = px.bar(g, x='Shipping Mode',
                  y=[PROB, 'Late_delivery_risk'],
                  barmode='group',
                  title='Shipping mode risk: predicted vs actual')
    st.plotly_chart(fig2, width='stretch')

else:
    st.subheader('Orders requiring immediate attention')
    sort_by = st.radio('Prioritise by',
        ['Highest probability', 'Most sales at risk'],
        horizontal=True)
    q = f[f['High']].copy()
    q['Sales at risk'] = (q['Sales'] * q[PROB]).round(2)
    key = PROB if sort_by == 'Highest probability' \
          else 'Sales at risk'
    q = q.sort_values(key, ascending=False)
    q.insert(0, 'Priority', range(1, len(q) + 1))
    st.metric('Orders needing attention', f'{len(q):,}')
    cols = ['Priority', 'Order Ref', 'Shipping Mode',
            'Market', 'Order Region', 'Customer Segment',
            'Sales', 'Sales at risk', PROB, 'Risk Category']
    st.dataframe(q[cols].head(500), width='stretch')
    st.download_button('Download High-Risk Order List (CSV)',
                       q[cols].to_csv(index=False),
                       'high_risk_orders.csv')
    st.info('Action: for First/Second Class orders, '
            're-prioritise picking and warn the customer.')
