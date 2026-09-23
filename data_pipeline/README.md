# Module 1 - Data Pipeline

## Overview

This module implements an end-to-end data pipeline using the Books to Scrape website.

The pipeline follows:

Scraping -> Cleaning -> Transformation -> SQLite Storage -> SQL Analysis -> Pandas Validation

## Data Collection

The first five catalogue pages of Books to Scrape are scraped, resulting in 100 books.

The extracted fields are:

- Title
- Price
- Star rating
- Availability
- Category

## Data Cleaning

The raw data is transformed into analysis-ready fields:

- `price_gbp` - numeric price in GBP
- `price_inr` - price converted to INR
- `rating` - integer rating from 1 to 5
- `in_stock` - Boolean stock status
- `category` - book category

The fixed conversion rate used is:

**1 GBP = 105.50 INR**

Numeric parsing failures are handled using median imputation.

## Database Design

The cleaned data is stored in a normalized SQLite database.

### categories table

- `category_id` - Primary Key
- `category_name` - Unique category name

### books table

- `book_id` - Primary Key
- `title` - Book title
- `price_gbp` - Price in GBP
- `price_inr` - Price in INR
- `rating` - Rating from 1 to 5
- `in_stock` - Availability status
- `category_id` - Foreign Key referencing `categories`

The `category_id` field connects the `books` table to the `categories` table.

## SQL Analysis

Six SQL queries are implemented:

1. SELECT + WHERE
2. ORDER BY + LIMIT
3. DISTINCT
4. BETWEEN
5. IN
6. JOIN

The SQL queries and their outputs are stored in:

`query_results.txt`

## Pandas and SQL Validation

The module uses `pd.read_sql()` to retrieve data from SQLite.

The SQL JOIN between `books` and `categories` is reproduced using `pd.merge()`.

The SQL JOIN and Pandas merge are compared to verify that both operations produce the same number of rows.

## Files

```text
data_pipeline/
├── 01_data_pipeline.ipynb
├── books.db
├── clean_books.csv
├── query_results.txt
└── README.md
