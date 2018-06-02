import pandas as pd
import datetime

comp = '../output/'
sub = pd.DataFrame()

sub1 = pd.read_csv(comp + 'blend01_2227.csv')
sub2 = pd.read_csv(comp + 'lgsub_2239.csv')
sub3 = pd.read_csv(comp + 'lgsub_2252.csv')
sub4 = pd.read_csv(comp + 'nn_226.csv')
sub5 = pd.read_csv(comp + 'Avito_Shanth_RNN_MIN_2260.csv')
sub6 = pd.read_csv(comp + 'Avito_Shanth_RNN_AVERAGE_2248.csv')

sub['item_id'] = sub1['item_id']

sub['deal_probability'] = (
    6.0 * sub1['deal_probability'] +
    2.0 * sub2['deal_probability'] +
    2.0 * sub6['deal_probability']
                          ) / 10.0

sub['deal_probability'].clip(0.0, 1.0, inplace=True)  # Between 0 and 1

now_time = datetime.datetime.now()
now_time = now_time.strftime('%Y-%m-%d-%H_%M_%S')

print("Finish sub merge...", now_time)
sub.to_csv(comp + 'sub_merge_%s.csv' % now_time, index=False)
