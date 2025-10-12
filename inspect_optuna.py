"""
Inspect Optuna Study Database
==============================
View trials, best parameters, and study progress.
"""

import optuna
import pandas as pd
from pathlib import Path
import json

def inspect_study(db_path: str, study_name: str = None):
    """
    Inspect an Optuna study from database file.
    
    Args:
        db_path: Path to optuna_study.db file
        study_name: Name of study (optional, will show all if None)
    """
    
    # Load study
    storage = f"sqlite:///{db_path}"
    
    print("="*80)
    print("OPTUNA STUDY INSPECTION")
    print("="*80)
    
    # List all studies in database
    print(f"\n📁 Database: {db_path}")
    print(f"Storage: {storage}\n")
    
    try:
        study_summaries = optuna.study.get_all_study_summaries(storage)
        print(f"Found {len(study_summaries)} study/studies:\n")
        
        for i, summary in enumerate(study_summaries):
            print(f"{i+1}. Study Name: {summary.study_name}")
            print(f"   Direction: {summary.direction}")
            print(f"   Trials: {summary.n_trials}")
            print(f"   Best Trial: {summary.best_trial.number if summary.best_trial else 'N/A'}")
            print()
        
        # If specific study requested, load it
        if study_name:
            target_study_name = study_name
        elif len(study_summaries) == 1:
            target_study_name = study_summaries[0].study_name
            print(f"Auto-selecting only study: {target_study_name}\n")
        else:
            print("Multiple studies found. Please specify study_name parameter.")
            print("Available studies:", [s.study_name for s in study_summaries])
            return None
        
        # Load the study
        study = optuna.load_study(study_name=target_study_name, storage=storage)
        
        print("="*80)
        print(f"STUDY: {target_study_name}")
        print("="*80)
        
        # Study info
        print(f"\n📊 Study Statistics:")
        print(f"   Total Trials: {len(study.trials)}")
        print(f"   Complete: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        print(f"   Pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        print(f"   Failed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
        print(f"   Running: {len([t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING])}")
        
        # Best trial
        if study.best_trial:
            print(f"\n🏆 Best Trial:")
            print(f"   Trial Number: {study.best_trial.number}")
            print(f"   Best Value: {study.best_value:.6f}")
            print(f"\n   Best Parameters:")
            for key, value in study.best_params.items():
                print(f"      {key}: {value}")
        
        # Trial history
        print(f"\n📈 Trial History:")
        trials_df = study.trials_dataframe()
        
        # Show last 10 trials
        print("\n   Last 10 Trials:")
        display_cols = ['number', 'state', 'value'] + [col for col in trials_df.columns if col.startswith('params_')]
        if 'value' in trials_df.columns:
            print(trials_df[display_cols].tail(10).to_string(index=False))
        else:
            print("   No completed trials yet.")
        
        # Save detailed report
        report_path = Path(db_path).parent / f"study_report_{target_study_name}.txt"
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"OPTUNA STUDY REPORT: {target_study_name}\n")
            f.write("="*80 + "\n\n")
            
            f.write("BEST PARAMETERS:\n")
            f.write("-"*80 + "\n")
            f.write(json.dumps(study.best_params, indent=2) + "\n\n")
            
            f.write("BEST VALUE:\n")
            f.write("-"*80 + "\n")
            f.write(f"{study.best_value:.6f}\n\n")
            
            f.write("ALL TRIALS:\n")
            f.write("-"*80 + "\n")
            f.write(trials_df.to_string() + "\n")
        
        print(f"\n💾 Saved detailed report to: {report_path}")
        
        # Save CSV
        csv_path = Path(db_path).parent / f"trials_{target_study_name}.csv"
        trials_df.to_csv(csv_path, index=False)
        print(f"💾 Saved trials CSV to: {csv_path}")
        
        return study
        
    except Exception as e:
        print(f"❌ Error inspecting study: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_optimization_history(study, save_path: str = None):
    """Plot optimization history."""
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
            plot_slice
        )
        import plotly
        
        # Optimization history
        fig1 = plot_optimization_history(study)
        if save_path:
            fig1.write_html(f"{save_path}_optimization_history.html")
            print(f"📊 Saved optimization history plot")
        else:
            fig1.show()
        
        # Parameter importance
        if len(study.trials) > 1:
            fig2 = plot_param_importances(study)
            if save_path:
                fig2.write_html(f"{save_path}_param_importance.html")
                print(f"📊 Saved parameter importance plot")
            else:
                fig2.show()
        
        # Parallel coordinate
        if len(study.trials) > 1:
            fig3 = plot_parallel_coordinate(study)
            if save_path:
                fig3.write_html(f"{save_path}_parallel_coordinate.html")
                print(f"📊 Saved parallel coordinate plot")
            else:
                fig3.show()
        
    except ImportError:
        print("⚠️  Install plotly for visualizations: pip install plotly")
    except Exception as e:
        print(f"⚠️  Could not create plots: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Inspect Optuna study database")
    parser.add_argument("--db", type=str, default="optuna_study.db", 
                       help="Path to Optuna database file")
    parser.add_argument("--study-name", type=str, default=None,
                       help="Name of study to inspect (optional)")
    parser.add_argument("--plot", action="store_true",
                       help="Generate visualization plots")
    
    args = parser.parse_args()
    
    # Inspect study
    study = inspect_study(args.db, args.study_name)
    
    # Generate plots if requested
    if args.plot and study:
        save_base = str(Path(args.db).parent / "study_plots")
        plot_optimization_history(study, save_base)