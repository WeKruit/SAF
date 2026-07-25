export function encodeAssetPath(path) {
  const bytes = new TextEncoder().encode(path);
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary)
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/g, "");
}

const SHA256_PATTERN = /^sha256:[0-9a-f]{64}$/;

async function readJson(response) {
  if (!response.ok) {
    const message =
      response.status === 404
        ? "Requested evidence asset was not found"
        : "Evidence asset could not be verified";
    throw new Error(message);
  }
  return response.json();
}

export class DashboardDataClient {
  constructor({ fetchImpl = fetch } = {}) {
    this.fetchImpl = fetchImpl;
    this.catalog = null;
    this.catalogPromise = null;
    this.assetCache = new Map();
    this.manifestSha256 = null;
    this.game = null;
    this.core = null;
    this.contracts = null;
    this.gameGeneration = 0;
    this.marketGeneration = 0;
    this.associationGeneration = 0;
  }

  async loadCatalog() {
    if (this.catalog) return this.catalog;
    this.catalogPromise ??= (async () => {
      const response = await this.fetchImpl("/api/dashboard/catalog");
      const manifestSha256 = response.headers.get(
        "x-dashboard-manifest-sha256",
      );
      if (!SHA256_PATTERN.test(manifestSha256 ?? "")) {
        throw new Error("Dashboard catalog has no verified manifest snapshot");
      }
      const catalog = await readJson(response);
      this.catalog = catalog;
      this.manifestSha256 = manifestSha256;
      return catalog;
    })();
    try {
      await this.catalogPromise;
    } catch (error) {
      this.catalogPromise = null;
      throw error;
    }
    return this.catalog;
  }

  async fetchAsset(reference, { signal } = {}) {
    const batch = encodeURIComponent(this.catalog.batch_id);
    const manifest = encodeURIComponent(this.manifestSha256);
    const cacheKey = `${this.manifestSha256}:${reference.relative_path}`;
    if (this.assetCache.has(cacheKey)) return this.assetCache.get(cacheKey);
    const response = await this.fetchImpl(
      `/api/dashboard/assets/${encodeAssetPath(
        reference.relative_path,
      )}?batch=${batch}&manifest=${manifest}`,
      { signal },
    );
    if (
      response.headers.get("x-dashboard-manifest-sha256") !==
      this.manifestSha256
    ) {
      throw new Error("Dashboard asset snapshot does not match its catalog");
    }
    const asset = await readJson(response);
    this.assetCache.set(cacheKey, asset);
    return asset;
  }

  async selectGame(gameId, { signal } = {}) {
    const generation = ++this.gameGeneration;
    const catalog = await this.loadCatalog();
    const game = catalog.payload.games.find((item) => item.game_id === gameId);
    if (!game) throw new Error(`Unknown game: ${gameId}`);
    const core = await this.fetchAsset(game.assets.core, { signal });
    if (generation !== this.gameGeneration) {
      throw new Error(`Game selection superseded: ${gameId}`);
    }
    this.game = game;
    this.core = core;
    this.contracts = null;
    return core;
  }

  async loadContracts({ signal } = {}) {
    if (!this.game) throw new Error("Select a game before loading contracts");
    if (this.contracts) return this.contracts;
    const game = this.game;
    const contracts = await this.fetchAsset(game.assets.contracts, { signal });
    if (this.game !== game) {
      throw new Error("Contract request superseded by another game");
    }
    this.contracts = contracts;
    return contracts;
  }

  async loadAssociations({ signal } = {}) {
    if (!this.game) throw new Error("Select a game before loading associations");
    const generation = ++this.associationGeneration;
    const game = this.game;
    const associations = await this.fetchAsset(game.assets.associations, {
      signal,
    });
    if (
      this.game !== game ||
      generation !== this.associationGeneration
    ) {
      throw new Error("Association request superseded by another game");
    }
    return associations;
  }

  async selectMarket(logicalMarketId, { signal } = {}) {
    if (!this.game) throw new Error("Select a game before selecting a market");
    if (!this.contracts) {
      throw new Error("Load contracts before selecting a market");
    }
    const generation = ++this.marketGeneration;
    const game = this.game;
    const contract = this.contracts.payload.contracts.find(
      (item) => item.logical_market_id === logicalMarketId,
    );
    if (!contract) throw new Error(`Unknown logical market: ${logicalMarketId}`);
    const market = await this.fetchAsset(contract.market_asset, { signal });
    if (this.game !== game || generation !== this.marketGeneration) {
      throw new Error("Market request superseded by another game");
    }
    return market;
  }
}
