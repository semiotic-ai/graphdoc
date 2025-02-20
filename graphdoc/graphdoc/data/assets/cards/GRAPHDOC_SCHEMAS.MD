---
language: en
---

# GraphDoc Schemas

This repository contains the schemas for the GraphDoc project. The schemas are used as training data for the GraphDoc pipeline, which is a GraphQL schema documentation process.

## Dataset

The dataset is a collection of parsed, or full, GraphQL schemas. The dataset complies with strong typing for categories, as is described in the table below.

| category (str) | rating (str) | schema_name (str) | schema_type (str) | schema_str (str) | 
|----------------|--------------|-------------------|-------------------|------------------|
| [ "perfect", "almost perfect", "poor but correct", "incorrect" ] | ["4", "3", "2", "1"] | str | ["full schema", "table schema"] | str |