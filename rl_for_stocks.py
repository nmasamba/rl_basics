import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like

import numpy as np
import datetime
import pandas_datareader.data as web
import random
from collections import defaultdict
import matplotlib.pyplot as plt

# PROBLEM DEFINITION
# Can we build a tool to trade stocks over the course of 10
# years by buying, selling or holding the S&P500 and 3-month T-Bills?

#tbill = web.DataReader('TB3MS', 'fred', '1934-01-01', '2018-11-29') # monthly t-bill dat from FRED
#tbill['Date'] = tbill.index 

tbill = web.DataReader('USTREASURY/BILLRATES', 'quandl', '2001-12-31', '2018-11-29') # monthly t-bill dat from Quandl
tbill = tbill[::-1]
#print('TBill', tbill)

sp = web.DataReader('MULTPL/SP500_REAL_PRICE_MONTH', 'quandl', '1871-01-01', '2018-11-29') # monthly S&P 500 data from Quandl
sp = sp[::-1]
#print('SP500', sp)



class QTrader(object):
	def __init__(self):
		self.stock_data = pd.merge(sp, tbill, on=['Date'])
		self.returns = pd.DataFrame({
			'stocks': self.stock_data['Value'].rolling(window=2).apply(lambda x: x[1] / x[0] - 1, raw='True'),
			#'tbills': self.stock_data['TB3MS'].rolling(window=2).apply(lambda x: x[1] / x[0] - 1, raw='True')
			'tbills': (self.stock_data['13WkBankDiscountRate'] / 100 + 1) ** (1/52) - 1,
			#'tbills': (self.stock_data['TB3MS'] / 100 + 1) ** (1/252) - 1,
			}, index = self.stock_data.index)
		
		self.returns['risk_adjusted'] = self.returns.stocks - self.returns.tbills
	

		self.returns['risk_adjusted_moving'] = self.returns.risk_adjusted.rolling(window=12).apply(lambda x: x.mean(), raw='True')
		self.returns['risk_adjusted_stdev'] = self.returns.risk_adjusted.rolling(window=12).apply(lambda x: x.std(), raw='True')
		self.returns['risk_adjusted_high'] = self.returns.risk_adjusted_moving + 1.5 * self.returns.risk_adjusted_stdev
		self.returns['risk_adjusted_low'] = self.returns.risk_adjusted_moving - 1.5 * self.returns.risk_adjusted_stdev
		# STATE
		# Given 12 weeks of data, what is the high average and what is the low average?
		# If the return is higher than high_average -> Buy (1)
		# If the return is lower than low average -> Sell (-1)
		# Else do nothing (0)
		# Std dev is used to determine higher/lower thresholds - this is the essence of Bollinger Bands!
		self.returns['state'] = (self.returns.risk_adjusted > self.returns.risk_adjusted_high).astype('int') - (self.returns.risk_adjusted < self.returns.risk_adjusted_low).astype('int')

	def buy_and_hold(self, dates):
		return pd.Series(1, index = dates)

	def buy_tbills(self, dates):
		return pd.Series(0, index = dates)

	def random(self, dates):
		return pd.Series(np.random.randint(-1, 2, size=len(dates)), index = dates)

	def evaluate(self, holdings):
		return pd.Series(self.returns.tbills + holdings * (self.returns.risk_adjusted) + 1, index = holdings.index).cumprod()

	def sharpe(self, holdings):
		returns = holdings * (self.returns.stocks - self.returns.tbills)
		return np.nanmean(returns) / np.nanstd(returns)


	def q_holdings(self, training_indexes, testing_indexes):
		factors = pd.DataFrame({'action':0, 'reward':0, 'state':0}, index = training_indexes)
		q = {0: {1:0, 0:0, -1:0}}

		for i in range(1000): #episodes
			last_row, last_date = None, None

			# Q Training

			for date, row in factors.iterrows():
				return_data = self.returns.loc[date]

				if return_data.state not in q:
					q[return_data.state] = {1:0, 0:0, -1:0}

				if last_row is None or np.isnan(return_data.state):
					state = 0
					reward = 0
					action = 0
				else:
					state = int(return_data.state)
					if random.random() > 0.01: # Every now and then, explore rather than exploit
						action = max(q[state], key=q[state].get)
					else:
						action = random.randint(-1,1)

					reward = last_row.action * (return_data.stocks - return_data.tbills) # Optimise for risk-free investments

					factors.loc[date, 'reward'] = reward
					factors.loc[date, 'action'] = action
					factors.loc[date, 'state'] = return_data.state

					# Learning params
					alpha = 0.2
					discount = 0.95

					update = alpha * (factors.loc[date, 'reward'] + discount * max(q[row.state].values()) - q[state][action])

					if not np.isnan(update):
						q[state][action] += update

				last_date, last_row = date, factors.loc[date]

			sharpe = self.sharpe(factors.action)

			if sharpe > 0.5:
				break

			print("For episode {} we get a Sharpe Ratio of {}".format(i, sharpe))


		# Q Testing
		testing = pd.DataFrame({'action':0, 'state':0}, index = testing_indexes)
		testing['state'] = self.returns.loc[testing_indexes, 'state']
		testing['action'] = testing['state'].apply(lambda state: max(q[state], key=q[state].get))

		return testing.action
	


	def graph_portfolio(self):
		midpoint = int(len(self.returns.index) / 2)
		training_indexes = self.returns.index[:midpoint]
		testing_indexes = self.returns.index[midpoint:]

		portfolios = pd.DataFrame({
			'buy_and_hold': self.evaluate(self.buy_and_hold(testing_indexes)),
			'buy_tbills': self.evaluate(self.buy_tbills(testing_indexes)),
			'random': self.evaluate(self.random(testing_indexes)),
			'qtrader': self.evaluate(self.q_holdings(training_indexes, testing_indexes))
			}, index = testing_indexes)

		portfolio_values = pd.DataFrame({
			'buy_and_hold': self.evaluate(portfolios.buy_and_hold),
			'buy_tbills': self.evaluate(portfolios.buy_tbills),
			'random': self.evaluate(portfolios.random),
			'qtrader': self.evaluate(portfolios.qtrader), 
			}, index = testing_indexes)

		portfolio_values.plot()

		plt.annotate("Buy and hold Sharpe Ratio: {} \n QTrader: {}".format(self.sharpe(portfolios.buy_and_hold), self.sharpe(portfolios.qtrader)), xy=(0.25, 0.95), xycoords='axes fraction')

		plt.show()

		# need sharpe methods

if __name__ == "__main__":
	q = QTrader()
	q.graph_portfolio()


