
import os
import argparse
import numpy
import math
import matplotlib.pyplot as plt
import pickle


def real_score(in_imp, fit_imp, type='Mean'):
    real_scale = 1./(0.0001+numpy.std(in_imp[:,0]))
    imag_scale = 1./(0.0001+numpy.std(in_imp[:, 1]))
    return (numpy.mean(
             (numpy.expand_dims(numpy.array([real_scale, imag_scale]), axis=0)*(in_imp - fit_imp))**2.  ))**(1./2.)

def complexity_score(params):
    num_zarcs = 3
    rs = params[2:2 + num_zarcs]

    l_half = numpy.square(numpy.sum(numpy.exp(.5 * rs)))
    l_1 = numpy.sum(numpy.exp(rs))
    complexity_loss = l_half/(1e-8 + l_1)

    return complexity_loss

def l1_norm(params):
    num_zarcs = 3
    rs = params[2:2 + num_zarcs]

    l_1 = numpy.sum(numpy.exp(rs))

    return l_1



def sample_of_fits(res_, filename):
    # Figure:
    # Samples of fits

    sorted_sorted_results = sorted(res_, key=lambda x: real_score(x[1], x[2], type='Mean'), reverse=True)

    list_to_print = sorted_sorted_results

    if not plot_all:
        list_of_indecies = [0.01, 0.05, 0.1, 0.25, .5, 0.9]
        list_of_subplots = [
            (2, 3, 1),
            (2, 3, 2),
            (2, 3, 3),
            (2, 3, 4),
            (2, 3, 5),
            (2, 3, 6),
        ]
    else:
        list_of_indecies = [3 * i for i in range(int(len(list_to_print) / 3))]

    fig = plt.figure(figsize=(14,9))
    for index, i_frac in enumerate(list_of_indecies):
        if not plot_all:
            i = int(i_frac * len(list_to_print))
        else:
            i = i_frac

        ax = fig.add_subplot(3, 2, 1 + index)
        fit_colors = ['r', 'g', 'b']
        param_colors = ['c', 'm', 'y']
        max_y = -100000.
        for k in range(3):
            j = i + k
            num_zarcs = 3
            wcs = list_to_print[j][3][2 + num_zarcs + 3: 2 + num_zarcs + 3 + num_zarcs]
            freqs = list_to_print[j][0]
            indecies = []
            for i_wc in range(num_zarcs):
                freqs_delta = [(x - wcs[i_wc]) ** 2 for x in freqs]
                indecies.append(min(range(len(freqs_delta)), key=freqs_delta.__getitem__))

            max_y = max(max_y, 100 * numpy.max(-list_to_print[j][1][:, 1]))
            ax.scatter(100 * list_to_print[j][1][:, 0], -100 * list_to_print[j][1][:, 1], c=fit_colors[k])

            ax.plot(100 * list_to_print[j][2][:, 0], -100 * list_to_print[j][2][:, 1], c=fit_colors[k])

            ax.scatter([100 * list_to_print[j][2][i_, 0] for i_ in indecies],
                       [-100 * list_to_print[j][2][i_, 1] for i_ in indecies],
                       c=fit_colors[k], marker="*", s=500,
                       label='R=({}, {}, {})'.format(
                           int(round(100. * math.exp(list_to_print[j][3][2]))),
                           int(round(100. * math.exp(list_to_print[j][3][3]))),
                           int(round(100. * math.exp(list_to_print[j][3][4]))))
                       )
        ax.set_ylim(bottom=-5, top=round(max_y) + 1)
        print('i: {}'.format(i))
        print('percentage: {}'.format(100. * float(i) / float(len(sorted_sorted_results))))
        ax.legend()
        ax.set_title('percentile: {}'.format(int(round(100. * float(i) / float(len(sorted_sorted_results))))))

    fig.tight_layout(h_pad=0., w_pad=0.)
    plt.savefig(filename)
    plt.close()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    #File paths
    parser.add_argument('--data_dir', default='RealData')
    parser.add_argument('--fra_no_finetune_file', default="results_of_inverse_model.file")
    parser.add_argument('--fra_finetune_file', default="results_fine_tuned_with_adam_{}.file")
    parser.add_argument('--eis_no_finetune_file', default="results_of_inverse_model_eis.file")
    parser.add_argument('--eis_finetune_file', default="results_eis_fine_tuned_with_adam_{}.file")
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--output_dir', default='OutputData')

    args = parser.parse_args()


    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)


    plot_all = False
    print_filenames = False


    with open(os.path.join(".", args.data_dir, args.fra_no_finetune_file), 'rb') as f:

        results_1 = pickle.load(f)
        print('got {} records fra no-finetune'.format(len(results_1)))
        res_ = results_1
        sample_of_fits(res_, os.path.join(args.output_dir, 'fra_no_finetune_samples.png') )
        scores_1 = sorted(map(lambda x: real_score(x[1], x[2], type='Mean'), results_1), reverse=True)
        scores_c_1 = sorted(map(lambda x: complexity_score(x[3]), results_1), reverse=True)
        scores_l_1 = sorted(map(lambda x: l1_norm(x[3]), results_1), reverse=True)


    with open(os.path.join(".", args.data_dir, args.eis_no_finetune_file), 'rb') as f:
        results_2 = pickle.load(f)
        sample_of_fits(results_2, os.path.join(args.output_dir, 'eis_no_finetune_samples.png'))
        print('got {} records eis no-finetune'.format(len(results_2)))


        scores_2 = sorted(map(lambda x: real_score(x[1], x[2], type='Mean'), results_2), reverse=True)
        scores_c_2 = sorted(map(lambda x: complexity_score(x[3]), results_2), reverse=True)
        scores_l_2 = sorted(map(lambda x: l1_norm(x[3]), results_2), reverse=True)



    scores_finetuned_adam = []
    scores_c_finetuned_adam = []
    scores_l_finetuned_adam = []
    steps_adam = [args.steps]
    styles = ['-']


    for i in steps_adam:
        with open(os.path.join(".", args.data_dir, args.fra_finetune_file.format(i)), 'rb') as f:

            dummy = pickle.load(f)

            sample_of_fits(dummy, os.path.join(args.output_dir, 'fra_finetune_samples.png'))
            print('got {} records fra finetune'.format(len(dummy)))

            scores_finetuned_adam.append( sorted(map(lambda x: real_score(x[1], x[2], type='Mean'), dummy), reverse=True))
            scores_c_finetuned_adam.append( sorted(map(lambda x: complexity_score(x[3]), dummy), reverse=True))
            scores_l_finetuned_adam.append(sorted(map(lambda x: l1_norm(x[3]), dummy), reverse=True))




    scores_finetuned_adam_eis = []
    scores_c_finetuned_adam_eis = []
    scores_l_finetuned_adam_eis = []
    steps_adam_eis = [args.steps]



    for i in steps_adam_eis:
        with open(os.path.join(".", args.data_dir, args.eis_finetune_file.format(i)), 'rb') as f:
            dummy = pickle.load(f)

            sample_of_fits(dummy, os.path.join(args.output_dir, 'eis_finetune_samples.png'))
            print('got {} records eis finetune'.format(len(dummy)))

            scores_finetuned_adam_eis.append( sorted(map(lambda x: real_score(x[1], x[2], type='Mean'), dummy), reverse=True))
            scores_c_finetuned_adam_eis.append( sorted(map(lambda x: complexity_score(x[3]), dummy), reverse=True))
            scores_l_finetuned_adam_eis.append(sorted(map(lambda x: l1_norm(x[3]), dummy), reverse=True))


















    #Figure:
    #progress in error during training


    fig = plt.figure()
    ax = fig.add_subplot(111)


    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.set_xlim(.5, 100.)
    ax.set_ylim(.01, .5)
    ax.plot([100.* float(x)/float(len(scores_1)) for x in range(len(scores_1))],
                scores_1, color='k',linestyle='--', linewidth=3, label='(FRA) inverse model')

    ax.plot([100.* float(x)/float(len(scores_2)) for x in range(len(scores_2))],
                scores_2, color='r',linestyle='--', linewidth=3, label='(EIS) inverse model')


    for index in range(len(steps_adam)):
        ax.plot([100. * float(x) / float(len(scores_c_finetuned_adam[index])) for x in range(len(scores_c_finetuned_adam[index]))],
            scores_finetuned_adam[index], color='k',linestyle=styles[index], linewidth=3, label='(FRA) inverse model + {} steps of finetuning'.format(steps_adam[index]))


    for index in range(len(steps_adam_eis)):
        ax.plot([100. * float(x) / float(len(scores_c_finetuned_adam_eis[index])) for x in range(len(scores_c_finetuned_adam_eis[index]))],
            scores_finetuned_adam_eis[index], color='r',linestyle=styles[index], linewidth=3, label='(EIS) inverse model + {} steps of finetuning'.format(steps_adam_eis[index]))





    plt.legend()
    plt.xlabel('percentile')
    plt.ylabel('MSE scaled by standard deviation')
    plt.savefig(os.path.join(args.output_dir, 'error_during_training.png'))
    plt.close()


    # Figure:
    # progress in complexity during training

    fig = plt.figure()
    ax = fig.add_subplot(111)




    ax.plot([100.* float(x)/float(len(scores_1)) for x in range(len(scores_1))],
                scores_c_1, color='k',linestyle='--', linewidth=3, label='(FRA) inverse model')

    ax.plot([100.* float(x)/float(len(scores_2)) for x in range(len(scores_2))],
                scores_c_2, color='r',linestyle='--', linewidth=3, label='(EIS) inverse model')

    for index in range(len(steps_adam)):
        ax.plot([100. * float(x) / float(len(scores_c_finetuned_adam[index])) for x in range(len(scores_c_finetuned_adam[index]))],
            scores_c_finetuned_adam[index], color='k',linestyle=styles[index], linewidth=3, label='(FRA) inverse model + {} steps of finetuning'.format(steps_adam[index]))

    for index in range(len(steps_adam_eis)):
        ax.plot([100. * float(x) / float(len(scores_c_finetuned_adam_eis[index])) for x in range(len(scores_c_finetuned_adam_eis[index]))],
            scores_c_finetuned_adam_eis[index], color='r',linestyle=styles[index], linewidth=3, label='(EIS) inverse model + {} steps of finetuning'.format(steps_adam_eis[index]))




    plt.legend()
    plt.xlabel('percentile')
    plt.ylabel('Zarc complexity')
    plt.savefig(os.path.join(args.output_dir, 'complexity_during_training.png'))
    plt.close()









    # Figure:
    # progress in L1 norm during training

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_yscale('log')
    ax.set_ylim(0.01,1)
    ax.plot([100.* float(x)/float(len(scores_1)) for x in range(len(scores_1))],
                scores_l_1, color='k',linestyle='--', linewidth=3, label='(FRA) inverse model')

    ax.plot([100.* float(x)/float(len(scores_2)) for x in range(len(scores_2))],
                scores_l_2, color='r',linestyle='--', label='(EIS) inverse model')



    for index in range(len(steps_adam)):
        ax.plot([100. * float(x) / float(len(scores_c_finetuned_adam[index])) for x in range(len(scores_c_finetuned_adam[index]))],
            scores_l_finetuned_adam[index], color='k',linestyle=styles[index], linewidth=3, label='(FRA) inverse model + {} steps of finetuning'.format(steps_adam[index]))

    for index in range(len(steps_adam_eis)):
        ax.plot([100. * float(x) / float(len(scores_c_finetuned_adam_eis[index])) for x in range(len(scores_c_finetuned_adam_eis[index]))],
            scores_l_finetuned_adam_eis[index], color='r',linestyle=styles[index], linewidth=3, label='(EIS) inverse model + {} steps of finetuning'.format(steps_adam_eis[index]))




    plt.legend()
    plt.xlabel('percentile')
    plt.ylabel('L1 norm')
    plt.savefig(os.path.join(args.output_dir, 'l1_during_training.png'))
    plt.close()
























