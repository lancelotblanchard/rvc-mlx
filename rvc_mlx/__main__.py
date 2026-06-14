"""Allow `python -m rvc_mlx ...` as an alias for the inference CLI."""

from rvc_mlx.infer import main

raise SystemExit(main())
