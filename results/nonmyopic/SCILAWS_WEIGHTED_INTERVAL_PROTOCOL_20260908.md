# Full weighted interval feasibility audit

Prospective zero-integration audit: all24 initialized fixtures, eight root actions,
all order4 branches, eight continuation actions. Compute existing analytic bounds
for every terminal action. Branch minimum lies between minimum lower and minimum
upper. Weight these by positive quadrature masses and add the same signed outer
correction to both endpoints. This bounds only the hybrid fixed-outer-quadrature
objective with exact terminal risk; outer integration error remains unbounded.

Allocate5e-5 midpoint terminal uncertainty, leaving5e-5 of the prior1e-4 total
accuracy tolerance for outer error, which still needs independent qualification.
Report every root width and the ideal minimum number of branch minima needing
exact refinement: remove the largest weighted widths until residual<=1e-4.
This is an optimistic work lower bound, not a runnable plan or runtime estimate;
making a branch minimum exact may require several terminal integrations.

At most24576 small analytic interval calculations, no nested numerical integration,
no source labels, no action/support filtering. Explicit one-thread BLAS. All cases
remain represented, irrespective of direction. No stage/deployment authorization.
