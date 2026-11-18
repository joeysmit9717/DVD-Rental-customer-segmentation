--return columns of 1) customer_id and 2) which category rented (number) and 3) fractions of categories per customer_id
WITH 
cust_cat_table AS (
	SELECT customer.customer_id AS CUS,  
	film_category.category_id AS CAT,
	name AS CAT_name
	FROM customer
	INNER JOIN rental ON
	customer.customer_id = rental.customer_id
	INNER JOIN payment ON
	customer.customer_id = payment.customer_id
	INNER JOIN inventory ON
	rental.inventory_id = inventory.inventory_id
	INNER JOIN film_category ON
	inventory.film_id = film_category.film_id
	INNER JOIN category ON
	film_category.category_id = category.category_id
	ORDER BY customer.customer_id
), 
--create the fraction of category per customer
cat_pct AS (
	SELECT 
	CUS, CAT,
	COUNT(*)::numeric / SUM(COUNT(*)) OVER (PARTITION BY CUS)
	AS CATEG_pct
	FROM cust_cat_table
	GROUP BY CUS, CAT
)


--restructure the fraction columns to a table suited for further data-analysis
SELECT
    cus AS customer_id,
	AVG(amount) as Average,
	SUM(amount) as Total,
    COALESCE(MAX(categ_pct) FILTER (WHERE cat = 1), 0)  AS Action,
    COALESCE(MAX(categ_pct) FILTER (WHERE cat = 2), 0)  AS Animation,
    COALESCE(MAX(categ_pct) FILTER (WHERE cat = 3), 0)  AS Children,
    COALESCE(MAX(categ_pct) FILTER (WHERE cat = 4), 0)  AS Classics,
    COALESCE(MAX(categ_pct) FILTER (WHERE cat = 5), 0)  AS Comedy,
    COALESCE(MAX(categ_pct) FILTER (WHERE cat = 6), 0)  AS Documentary,
    COALESCE(MAX(categ_pct) FILTER (WHERE cat = 7), 0)  AS Drama,
    COALESCE(MAX(categ_pct) FILTER (WHERE cat = 8), 0)  AS Family,
    COALESCE(MAX(categ_pct) FILTER (WHERE cat = 9), 0)  AS Foreign,
    COALESCE(MAX(categ_pct) FILTER (WHERE cat = 10), 0) AS Games,
    COALESCE(MAX(categ_pct) FILTER (WHERE cat = 11), 0) AS Horror,
    COALESCE(MAX(categ_pct) FILTER (WHERE cat = 12), 0) AS Music,
    COALESCE(MAX(categ_pct) FILTER (WHERE cat = 13), 0) AS New,
    COALESCE(MAX(categ_pct) FILTER (WHERE cat = 14), 0) AS Scifi,
    COALESCE(MAX(categ_pct) FILTER (WHERE cat = 15), 0) AS Sports,
    COALESCE(MAX(categ_pct) FILTER (WHERE cat = 16), 0) AS Travel
FROM cat_pct 
INNER JOIN payment ON
cat_pct.CUS = payment.customer_id
GROUP BY cus
ORDER BY cus;


