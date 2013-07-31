SELECT uopen.cached_category_list AS category_list_opener,
       uview.cached_category_list AS category_list_viewer,
       uopen.cached_interest_list AS interest_list_opener,
       uview.cached_interest_list AS interest_list_viewer,
       pinfo.cached_category_list AS promo_category_list,
       pinfo.cached_interest_list AS promo_interest_list,
       pa.promotion_id,
       pa.status,
       pinfo.created_at,
       uopen.total_fans AS total_fans_opener,
       uview.total_fans AS total_fans_viewer,
       uopen.id AS user_id_opener,
       uview.id AS user_id_viewer,
       uopen.zipcode AS zipcode_opener,
       uview.zipcode AS zipcode_viewer

FROM (SELECT promotion_id, user_id AS viewer_id, status
     FROM promotion_approvals 
     WHERE promotion_id > 200600 AND promotion_id < 200800) AS pa
LEFT JOIN (
     SELECT id AS promotion_id, 
	    opener_id, 
	    created_at,
	    cached_category_list,
	    cached_interest_list
     FROM promotions
     WHERE id > 200600 AND id < 200800) AS pinfo
ON pa.promotion_id = pinfo.promotion_id
LEFT JOIN users AS uopen
ON pinfo.opener_id = uopen.id
LEFT JOIN users AS uview
ON pa.viewer_id = uview.id
ORDER BY promotion_id, user_id_viewer