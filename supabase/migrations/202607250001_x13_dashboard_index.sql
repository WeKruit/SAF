create table dashboard_batches (
  batch_id text primary key,
  experiment_id text not null,
  status text not null,
  manifest_sha256 text not null unique,
  builder_version text not null,
  claim_boundary jsonb not null,
  published_at timestamptz not null
);

create table dashboard_games (
  batch_id text not null references dashboard_batches(batch_id),
  game_id text not null,
  away_team text not null,
  home_team text not null,
  away_score integer not null,
  home_score integer not null,
  event_count integer not null,
  episode_count integer not null,
  contract_count integer not null,
  audit_status text not null,
  primary key (batch_id, game_id)
);

create table dashboard_assets (
  batch_id text not null references dashboard_batches(batch_id),
  asset_path text not null,
  game_id text,
  role text not null,
  media_type text not null,
  schema_version text not null,
  source_sha256 text not null,
  object_sha256 text not null,
  byte_length bigint not null,
  primary key (batch_id, asset_path)
);

alter table dashboard_batches enable row level security;
alter table dashboard_games enable row level security;
alter table dashboard_assets enable row level security;
