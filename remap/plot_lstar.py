import pandas
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tempfile


## Generates plots that do not use Type 3 fonts.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def plot_histogram(observations_array, bin_size=10, xlabel="X Axis", ylabel="Count", title="Default Title", output_file=None):
    counts, bins = np.histogram(observations_array)
    plt.hist(bins[:-1], bins, weights=counts)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if output_file is None:
        with tempfile.NamedTemporaryFile(mode="wb",delete=False) as f:
            plt.savefig(f, bbox_inches="tight")
            plt.close()
    else:
        plt.savefig(output_file, bbox_inches="tight")
        plt.close()

def plot_violin(data, xcol, ycol, xlabel="X Axis", ylabel="Y Axis", title="Default Title", output_file=None):
    unique_x_values = data[xcol].unique()
    violin_data = list(data.loc[data[xcol] == x, ycol].to_numpy() for x in unique_x_values)
    ## NOTE: Each element of data should be a frequency distribution of y values for a given x value
    ## For example, for "alphabet size, we only have 3 alphabet sizes.
    ## Compute mean, median, min, max, etc for each
    plt.violinplot(violin_data, positions=unique_x_values, showmeans=True, vert=True, showextrema=True, widths=1)
    #plt.xticks([y+1 for y in range(len(unique_x_values))], labels=unique_x_values) 
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.gca().set_box_aspect(3/4)
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

def plot_regret(data_group, xlabel="X Axis", ylabel="Y Axis", title="Default Title", output_file=None):
    for idx, data in enumerate(data_group):
        c = f"C{idx}"
        X = data["'Variable'"].to_numpy()
        Y = data["'Avg Regret'"].to_numpy()
        variance = data["'Variance'"].to_numpy()
        std = np.sqrt(variance)
        plt.fill_between(X, Y-std, Y+std, alpha=0.5, linewidth=0, color=c)
        plt.plot(X, Y, alpha=1.0, linewidth=3, color=c)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.gca().set_box_aspect(9/14)
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

def plot_scatter(x, y, xlabel="X Axis", ylabel="Y Axis", title="Default Title", output_file=None):
    #max_x = list(max(eval(e)) for e in x)
    #plt.scatter(max_x,y)
    plt.scatter(x,y)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.gca().set_box_aspect(3/4)

    if output_file is None:
        with tempfile.NamedTemporaryFile(mode="wb",delete=False) as f:
            plt.savefig(f, bbox_inches="tight")
            plt.close()
    else:
        plt.savefig(output_file, bbox_inches="tight")
        plt.close()

def plot_colored_scatter(data_group, xcol, ycol, xlabel="X Axis", ylabel="Y Axis", title="Default Title", output_file=None):
    ## data_group is a list of dataframes, each dataframe belongs to a category, i.e. specific task
    for idx in range(len(data_group)):
        data = data_group[idx]
        x = data[xcol]
        max_x = list(max(eval(e)) for e in x)
        y = data[ycol].to_numpy()
        plt.scatter(max_x,y,c=f"C{idx+4}",alpha=0.5)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.gca().set_box_aspect(3/4)
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

def read_csv(file_path, sep="#"):
    data = pandas.read_csv(file_path, sep=sep)
    return data

def parse_args():
    parser = argparse.ArgumentParser(description='Provide Filenames')
    parser.add_argument('--xlabel', type=str,
                        help='Label the x-axis')
    parser.add_argument('--ylabel', type=str,
                        help='Label the y-axis')
    parser.add_argument('--title', type=str,
                        help='Title of Plot')
    parser.add_argument('--save', type=str,
                        help='Name of the file to save under')
    parser.add_argument('--type', type=str, choices=["histogram", "violin_plot",
                        "plot", "box_plot_sequence", "comparison_plot",
                        "colored_scatter", "regret_plot",
                        "isomorphism_plot", "event_plot", "event_taxi_histogram"],
                        help='Type of figure to generate')
    parser.add_argument('--column', type=str, action="append",
                        help='Specify a column for each column you want plotted')
    parser.add_argument('--fprefix', type=str,
                        help='Prefix of the file')
    parser.add_argument('--fsuffix', type=str,
                        help='Suffix of the file')
    parser.add_argument('--fiter', type=int, action="append",
                        help='Integer iteration')
    parser.add_argument('--csv', type=str, action="append", default=None,
                        help='Specifies the CSV file path to read in')
    parser.add_argument('--groups', type=int, default=1,
                        help='Divides CSV files into groups')
    parser.add_argument('--expids', type=str, action="append",
                        help="Exp String ID for each experiment to be plotted")
    parser.add_argument('--legend', type=str, action="append",
                        help="Legend Label for each experiment to be plotted")
    args = parser.parse_args()
    return args

def main():
    plt.rcParams.update({'font.size': 20})
    args = parse_args()
    data = None
    data_groups = None
    if args.csv is not None:
        dfs = list(read_csv(csv) for csv in args.csv)
        ## Do round robin grouping
        ## If args.groups == 1, then every 1 belongs to the same group
        ## If args.groups == 10, then every 10th belongs to the same group
        data = pandas.concat(dfs)
        num_dfs = len(dfs)
        items_per_group = num_dfs // args.groups
        if args.groups > 1:
            data_groups = list()
            for group_id in range(args.groups):
                grouping = list(None for _ in range(items_per_group))
                print(f"Grouping {group_id}:")
                for g in range(items_per_group):
                    idx = g*args.groups + group_id
                    grouping[g] = dfs[idx]
                    print(f"    {args.csv[idx]}")
                data_groups.append(pandas.concat(grouping))
    if args.type == "histogram":
        #data = read_csv(args.csv)
        plot_histogram(data[args.column[0]].to_numpy(), xlabel=args.xlabel, ylabel=args.ylabel, title=args.title, output_file=args.save)
    elif args.type == "violin_plot":
        plot_violin(data, args.column[0], args.column[1], xlabel=args.xlabel, ylabel=args.ylabel, title=args.title, output_file=args.save)
    elif args.type == "regret_plot":
        plot_regret(data_groups, xlabel=args.xlabel, ylabel=args.ylabel, title=args.title, output_file=args.save)
    elif args.type == "colored_scatter":
        plot_colored_scatter(data_groups, args.column[0], args.column[1], xlabel=args.xlabel, ylabel=args.ylabel, title=args.title, output_file=args.save)
    elif args.type == "plot":
        #data = read_csv(args.csv)
        plot_scatter(data[args.column[0]].to_numpy(), data[args.column[1]].to_numpy(), xlabel=args.xlabel, ylabel=args.ylabel, title=args.title, output_file=args.save)
    elif args.type == "box_plot_sequence":
        plot_box_plot_sequence(args.fprefix, args.fsuffix, args.fiter, args.column[0], xlabel=args.xlabel, ylabel=args.ylabel, title=args.title, output_file=args.save)
    elif args.type == "comparison_plot":
        plot_comparison(args.fprefix, args.fsuffix, args.expids, args.fiter, args.column[0], args.legend, xlabel=args.xlabel, ylabel=args.ylabel, title=args.title, output_file=args.save)
    elif args.type == "isomorphism_plot":
        plot_isomorphism(args.fprefix, args.fsuffix, args.expids, args.fiter, args.column[0], args.legend, xlabel=args.xlabel, ylabel=args.ylabel, title=args.title, output_file=args.save)
    elif args.type == "event_plot":
        #data = read_csv(args.csv)
        plot_events(data, xlabel=args.xlabel, ylabel=args.ylabel, title=args.title, output_file=args.save)
    elif args.type == "event_taxi_histogram":
        #data = read_csv(args.csv)
        plot_event_taxi_distance(data["'Events'"], xlabel=args.xlabel, ylabel=args.ylabel, title=args.title, output_file=args.save)
    else:
        print("Hello")

def plot_event_taxi_distance(events_df, xlabel="X Axis", ylabel="Y Axis", title="Default Title", output_file=None):
    event_list_list = list(eval(e) for e in events_df)

    filtered_list_list = []
    for event_list in event_list_list:
        filtered_list = list(e for e in event_list if e[1] in ("initialization", "equivalence"))
        filtered_list_list.append(filtered_list)

    ## We will plot the delta between hypotheses
    print(filtered_list_list)

    taxi_dist = []

    for filtered_event_list in filtered_list_list:
        for idx in range(1, len(filtered_event_list)):
            _,_,n0,v0 = filtered_event_list[idx-1]
            _,_,n1,v1 = filtered_event_list[idx]
            taxi_dist.append((n1-n0) + (v1-v0))

    print(taxi_dist)

    plot_histogram(taxi_dist, bin_size=1, xlabel=xlabel, ylabel=ylabel, title=title, output_file=output_file)
    

def plot_events(data, xlabel="X Axis", ylabel="Y Axis", title="Default Title", output_file=None):
    events_df = data["'Events'"]
    event_list = list(eval(e) for e in events_df)
    max_number_of_states = data["'Number of States'"].max()
    max_number_of_representatives = data["'Num ECs'"].max()

    ## Put number of states on Y axis, number of values on X axis

    for trial in event_list:
        X = list(num_known_values for event_id, event_type, num_state, num_known_values in trial)
        Y = list(num_state for event_id, event_type, num_state, num_known_values in trial)

        plt.plot(X, Y, linewidth=1, color="grey", alpha=0.50)
        
        for event_id, event_type, num_state, num_known_values in trial:
            if event_type == "closure":
                plt.scatter(num_known_values, num_state, marker="o", s=100, color="blue")
            if event_type == "consistency":
                plt.scatter(num_known_values, num_state, marker="s", s=100, color="orange")
            if event_type == "termination":
                plt.scatter(num_known_values, num_state, marker="*", s=100, color="red")

        for event_id, event_type, num_state, num_known_values in trial:
            if event_type == "equivalence":
                plt.scatter(num_known_values, num_state, marker="x", s=50, color="green")
    plt.scatter(max_number_of_representatives, max_number_of_states, marker="*", s=100, color="red")
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.gca().set_box_aspect(3/4)

    if output_file is None:
        with tempfile.NamedTemporaryFile(mode="wb",delete=False) as f:
            plt.savefig(f, bbox_inches="tight")
            plt.close()
    else:
        plt.savefig(output_file, bbox_inches="tight")
        plt.close()


def plot_comparison(prefix, suffix, exp_ids, iter_list, target, legend, xlabel="X Axis", ylabel="Y Axis", title="Default Title", output_file=None):
    
    ## First read in all the datasets
    df = dict()
    for exp in exp_ids:
        df[exp] = dict()
        for x in iter_list:
            df[exp][x] = read_csv(f"{prefix}{exp}{x}{suffix}")

    X = iter_list
    #Y_means = list(list(df[exp][x][target].mean() for x in iter_list) for exp in exp_ids)
    #Y_stds = list(list(df[exp][x][target].std() for x in iter_list) for exp in exp_ids)
    #Y_upper = list()
    #Y_lower = list()
    #for idx in range(len(Y_means)):
    #    means_list = Y_means[idx]
    #    stds_list = Y_stds[idx]
    #    upper_list = list(means_list[jdx] + 1.96*stds_list[jdx] for jdx in range(len(means_list)))
    #    lower_list = list(means_list[jdx] - 1.96*stds_list[jdx] for jdx in range(len(means_list)))
    #    Y_upper.append(upper_list)
    #    Y_lower.append(lower_list)
    Y_medians = list(list(df[exp][x][target].median() for x in iter_list) for exp in exp_ids)
    # Quantiles: [0.2, 0.8]
    Y_lower_quantile = list(list(df[exp][x][target].quantile(0.2) for x in iter_list) for exp in exp_ids)
    Y_higher_quantile = list(list(df[exp][x][target].quantile(0.8) for x in iter_list) for exp in exp_ids)

    for idx, exp in enumerate(exp_ids):
        c = f"C{idx}"
        #plt.fill_between(X, Y_higher_quantile[idx], Y_lower_quantile[idx], alpha=0.5, linewidth=0, color=c)
        plt.fill_between(X, Y_lower_quantile[idx], Y_higher_quantile[idx], alpha=0.5, linewidth=0, color=c)
        #plt.fill_between(X, Y_lower[idx], Y_upper[idx], alpha=0.5, linewidth=0, color=c)
        plt.plot(X, Y_medians[idx], alpha=1.0, linewidth=1, color=c, label=legend[idx])
        #plt.plot(X, Y_means[idx], alpha=1.0, linewidth=2, color=c, label=legend[idx])
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    #plt.legend()
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

def plot_isomorphism(prefix, suffix, exp_ids, iter_list, target, legend, xlabel="X Axis", ylabel="Y Axis", title="Default Title", output_file=None):
    
    ## First read in all the datasets
    df = dict()
    for exp in exp_ids:
        df[exp] = dict()
        for x in iter_list:
            df[exp][x] = read_csv(f"{prefix}{exp}{x}{suffix}")

    X = iter_list
    Y_means = list(list(df[exp][x][target].ge(1).value_counts()[True]/df[exp][x][target].count() if True in df[exp][x][target].ge(1).value_counts() else 0.0 for x in iter_list) for exp in exp_ids)

    for idx, exp in enumerate(exp_ids):
        c = f"C{idx}"
        #plt.fill_between(X, Y_lower_quantile[idx], Y_higher_quantile[idx], alpha=0.5, linewidth=0, color=c)
        #plt.plot(X, Y_medians[idx], alpha=0.7, linewidth=1, color=c)
        plt.plot(X, Y_means[idx], alpha=1.0, linewidth=3, color=c, label=legend[idx])
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    #plt.legend()
    plt.grid()
    plt.gca().set_box_aspect(9/14)
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

def plot_box_plot_sequence(prefix, suffix, iter_list, target, xlabel="X Axis", ylabel="Y Axis", title="Default Title", output_file=None):
    ## First read in all the datasets
    df = dict()
    for x in iter_list:
        df[x] = read_csv(f"{prefix}{x}{suffix}")

    ## Compute mean, median, min, max, etc for each
    data = list(df[x][target].to_numpy() for x in iter_list)
    plt.violinplot(data, showmeans=True, showmedians=True, vert=True, showextrema=True)
   
    plt.xticks([y+1 for y in range(len(iter_list))], labels=iter_list) 
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()
    

if __name__ == "__main__":
    main()
