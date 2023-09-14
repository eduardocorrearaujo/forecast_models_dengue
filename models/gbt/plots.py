

import matplotlib.pyplot as plt  

def plot_prediction(city, dates, target, preds, preds25, preds975, train_size, doenca, label):
            
    fig, ax = plt.subplots(1,2, figsize = (10, 3.5))

    ax[0].plot(dates[:train_size], target[:train_size], label ='Data', color = 'black')

    ax[0].plot(dates[:train_size], preds[:train_size], label = 'RF model', color= 'tab:orange')

    ax[0].fill_between(dates[:train_size], preds25[:train_size],preds975[:train_size], alpha = 0.2, color= 'tab:orange')

    ax[0].set_title(f'Train set - {city}')
    ax[0].set_xlabel('Date')
    ax[0].set_ylabel('New cases')

    ax[0].grid()


    ax[1].plot(dates[train_size:], target[train_size:], label ='Data', color = 'black')

    ax[1].plot(dates[train_size:], preds[train_size:], label = 'RF model', color= 'tab:orange')

    ax[1].fill_between(dates[train_size:], preds25[train_size:],preds975[train_size:], alpha = 0.2, color= 'tab:orange')

    ax[1].grid()

    ax[1].set_title(f'Test set - {city}')
    ax[1].set_xlabel('Date')
    ax[1].set_ylabel('New cases')

    for tick in ax[1].get_xticklabels():
        tick.set_rotation(25)

    plt.savefig(f'./plots/{doenca}_{city}_{label}_train_test.png', dpi = 300, bbox_inches = 'tight')

    plt.show()

    return 

