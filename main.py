#!/usr/bin/env python3
"""
Value Investment Tool - CLI Interface

A stock screening and analysis tool for value investors.

Features:
- Criteria-based stock screening for US and SG markets
- Automated valuation using EPV vs Market Cap
- Financial anomaly detection

Usage:
    python main.py screen --market US --roe 15 --gross-margin 20
    python main.py screen --market SG --config criteria.json
    python main.py analyze --symbol DDI
    python main.py full --market US --roe 15 --output results.csv
"""
import json
import sys
from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from tabulate import tabulate

from data_loader import DataLoader
from screener import StockScreener
from valuation import Valuator
from anomaly_detector import AnomalyDetector

console = Console()


def load_config(config_path: str) -> dict:
    """Load criteria from a JSON config file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_results_table(df: pd.DataFrame, title: str = "Screening Results") -> Table:
    """Create a rich table from DataFrame."""
    table = Table(title=title, show_header=True, header_style="bold cyan")

    # Add columns
    for col in df.columns:
        table.add_column(col, overflow="fold")

    # Add rows
    for _, row in df.head(50).iterrows():  # Limit to 50 for display
        values = []
        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                if pd.isna(val):
                    values.append("-")
                elif 'Ratio' in col or 'Margin' in col or '%' in col:
                    values.append(f"{val:.2f}")
                else:
                    values.append(f"{val:.2f}")
            else:
                values.append(str(val) if not pd.isna(val) else "-")
        table.add_row(*values)

    return table


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """Value Investment Tool - Stock Screening and Analysis"""
    pass


@cli.command()
@click.option('--market', '-m', type=click.Choice(['US', 'SG', 'us', 'sg']),
              default='US', help='Market to screen (US or SG)')
@click.option('--gross-margin', type=float, help='Minimum Gross Margin %')
@click.option('--net-margin', type=float, help='Minimum Net Margin %')
@click.option('--roa', type=float, help='Minimum Return on Assets %')
@click.option('--roe', type=float, help='Minimum Return on Equity %')
@click.option('--revenue-growth', type=float, help='Minimum 5-Year Revenue Growth Rate')
@click.option('--eps-growth', type=float, help='Minimum 5-Year EPS Growth Rate')
@click.option('--debt-equity', type=float, help='Maximum Debt-to-Equity Ratio')
@click.option('--fcf-margin', type=float, help='Minimum Free Cash Flow Margin %')
@click.option('--roic-wacc', type=float, help='Minimum ROIC minus WACC')
@click.option('--rote-wacc', type=float, help='Minimum ROTE minus WACC')
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to criteria config file')
@click.option('--output', '-o', type=click.Path(), help='Output file path (CSV or JSON)')
@click.option('--valuation/--no-valuation', default=True, help='Include valuation classification')
@click.option('--limit', type=int, default=50, help='Maximum number of results to display')
def screen(market, gross_margin, net_margin, roa, roe, revenue_growth, eps_growth,
           debt_equity, fcf_margin, roic_wacc, rote_wacc, config, output, valuation, limit):
    """
    Screen stocks based on fundamental criteria.

    Examples:
        python main.py screen --market US --roe 15 --gross-margin 30
        python main.py screen --market SG --config config/criteria.json
        python main.py screen --market US --roe 15 --output results.csv
    """
    # Build criteria dictionary
    criteria = {}

    # Load from config if provided
    if config:
        config_criteria = load_config(config)
        criteria.update(config_criteria)
        console.print(f"[green]Loaded criteria from {config}[/green]")

    # Override with command-line options
    cli_criteria = {
        'gross_margin': gross_margin,
        'net_margin': net_margin,
        'roa': roa,
        'roe': roe,
        'revenue_growth_5y': revenue_growth,
        'eps_growth_5y': eps_growth,
        'debt_to_equity': debt_equity,
        'fcf_margin': fcf_margin,
        'roic_wacc': roic_wacc,
        'rote_wacc': rote_wacc
    }

    for key, value in cli_criteria.items():
        if value is not None:
            criteria[key] = value

    if not criteria:
        console.print("[yellow]No criteria specified. Showing all stocks with valuation.[/yellow]")

    # Run screening
    console.print(f"\n[bold]Screening {market.upper()} stocks...[/bold]")
    if criteria:
        console.print(f"Criteria: {criteria}")

    try:
        screener = StockScreener()
        results = screener.screen(
            market=market.upper(),
            criteria=criteria if criteria else None,
            include_valuation=valuation
        )

        console.print(f"\n[green]Found {len(results)} stocks matching criteria[/green]")

        if len(results) == 0:
            console.print("[yellow]No stocks match the specified criteria. Try relaxing the filters.[/yellow]")
            return

        # Format results for display
        formatted = screener.format_results(results)

        # Display summary
        stats = screener.get_summary_stats(results)
        if 'by_valuation' in stats and stats['by_valuation']:
            console.print("\n[bold]Valuation Summary:[/bold]")
            for val, count in stats['by_valuation'].items():
                console.print(f"  {val}: {count}")

        # Display table
        console.print()
        table = create_results_table(formatted.head(limit), f"Top {min(limit, len(formatted))} Results")
        console.print(table)

        if len(formatted) > limit:
            console.print(f"\n[dim]... and {len(formatted) - limit} more results[/dim]")

        # Export if output specified
        if output:
            output_path = Path(output)
            if output_path.suffix.lower() == '.json':
                results.to_json(output, orient='records', indent=2)
            else:
                results.to_csv(output, index=False)
            console.print(f"\n[green]Results exported to {output}[/green]")

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error during screening: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('symbol')
@click.option('--output', '-o', type=click.Path(), help='Output file path for report')
def analyze(symbol, output):
    """
    Analyze a company for financial anomalies.

    SYMBOL is the stock ticker (e.g., DDI, TEX, CHCI)

    Examples:
        python main.py analyze DDI
        python main.py analyze TEX --output tex_report.txt
    """
    console.print(f"\n[bold]Analyzing {symbol.upper()} for anomalies...[/bold]\n")

    try:
        detector = AnomalyDetector()
        report = detector.analyze(symbol)
        formatted_report = detector.format_report(report)

        # Display report
        console.print(formatted_report)

        # Export if output specified
        if output:
            with open(output, 'w') as f:
                f.write(formatted_report)
            console.print(f"\n[green]Report exported to {output}[/green]")

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        available = DataLoader().get_available_symbols()
        if available:
            console.print(f"[yellow]Available symbols: {', '.join(available)}[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error during analysis: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--market', '-m', type=click.Choice(['US', 'SG', 'us', 'sg']),
              default='US', help='Market to screen')
@click.option('--gross-margin', type=float, help='Minimum Gross Margin %')
@click.option('--net-margin', type=float, help='Minimum Net Margin %')
@click.option('--roa', type=float, help='Minimum Return on Assets %')
@click.option('--roe', type=float, help='Minimum Return on Equity %')
@click.option('--revenue-growth', type=float, help='Minimum 5-Year Revenue Growth Rate')
@click.option('--eps-growth', type=float, help='Minimum 5-Year EPS Growth Rate')
@click.option('--debt-equity', type=float, help='Maximum Debt-to-Equity Ratio')
@click.option('--fcf-margin', type=float, help='Minimum Free Cash Flow Margin %')
@click.option('--roic-wacc', type=float, help='Minimum ROIC minus WACC')
@click.option('--rote-wacc', type=float, help='Minimum ROTE minus WACC')
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to criteria config file')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--undervalued-only', is_flag=True, help='Only show undervalued stocks')
def full(market, gross_margin, net_margin, roa, roe, revenue_growth, eps_growth,
         debt_equity, fcf_margin, roic_wacc, rote_wacc, config, output, undervalued_only):
    """
    Run full pipeline: screen + valuation analysis.

    Examples:
        python main.py full --market US --roe 15 --undervalued-only
        python main.py full --market SG --config criteria.json --output results.csv
    """
    # Build criteria
    criteria = {}

    if config:
        config_criteria = load_config(config)
        criteria.update(config_criteria)

    cli_criteria = {
        'gross_margin': gross_margin,
        'net_margin': net_margin,
        'roa': roa,
        'roe': roe,
        'revenue_growth_5y': revenue_growth,
        'eps_growth_5y': eps_growth,
        'debt_to_equity': debt_equity,
        'fcf_margin': fcf_margin,
        'roic_wacc': roic_wacc,
        'rote_wacc': rote_wacc
    }

    for key, value in cli_criteria.items():
        if value is not None:
            criteria[key] = value

    # Run screening
    console.print(f"\n[bold]Running full analysis on {market.upper()} market...[/bold]")

    try:
        screener = StockScreener()
        results = screener.screen(
            market=market.upper(),
            criteria=criteria if criteria else None,
            include_valuation=True
        )

        console.print(f"[green]Screened {len(results)} stocks[/green]")

        # Apply undervalued filter if requested
        if undervalued_only:
            valuator = Valuator()
            results = valuator.get_undervalued(results)
            console.print(f"[green]Found {len(results)} undervalued stocks[/green]")

        if len(results) == 0:
            console.print("[yellow]No stocks match the criteria.[/yellow]")
            return

        # Get summary
        valuator = Valuator()
        summary = valuator.get_valuation_summary(results)

        console.print("\n[bold]Valuation Distribution:[/bold]")
        for status, count in summary['counts'].items():
            pct = summary['percentages'][status]
            console.print(f"  {status}: {count} ({pct}%)")

        if 'avg_margin_undervalued' in summary:
            console.print(f"\n  Avg Margin of Safety (Undervalued): {summary['avg_margin_undervalued']:.1f}%")

        # Display top results
        formatted = screener.format_results(results)
        console.print()
        table = create_results_table(formatted.head(30), "Top 30 Results")
        console.print(table)

        # Export
        if output:
            output_path = Path(output)
            if output_path.suffix.lower() == '.json':
                results.to_json(output, orient='records', indent=2)
            else:
                results.to_csv(output, index=False)
            console.print(f"\n[green]Results exported to {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
def criteria():
    """Show available screening criteria."""
    screener = StockScreener()
    criteria = screener.get_available_criteria()

    console.print("\n[bold]Available Screening Criteria:[/bold]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Parameter", style="green")
    table.add_column("Description")
    table.add_column("CLI Option", style="yellow")

    cli_options = {
        'gross_margin': '--gross-margin',
        'net_margin': '--net-margin',
        'roa': '--roa',
        'roe': '--roe',
        'revenue_growth_5y': '--revenue-growth',
        'eps_growth_5y': '--eps-growth',
        'debt_to_equity': '--debt-equity',
        'fcf_margin': '--fcf-margin',
        'roic_wacc': '--roic-wacc',
        'rote_wacc': '--rote-wacc'
    }

    for name, desc in criteria.items():
        table.add_row(name, desc, cli_options.get(name, ''))

    console.print(table)


@cli.command()
def symbols():
    """Show available symbols for anomaly analysis."""
    loader = DataLoader()
    available = loader.get_available_symbols()

    if available:
        console.print("\n[bold]Available symbols for anomaly analysis:[/bold]")
        for symbol in available:
            console.print(f"  - {symbol}")
    else:
        console.print("[yellow]No anomaly data files found.[/yellow]")


if __name__ == '__main__':
    cli()
