// Minimal structured logger. Stays out of the way of `docker compose logs`.

function fmt(level, args) {
  const ts = new Date().toISOString();
  const parts = args.map((a) =>
    a instanceof Error
      ? `${a.message}\n${a.stack}`
      : typeof a === 'object'
        ? JSON.stringify(a)
        : String(a),
  );
  return `${ts} ${level} recorder | ${parts.join(' ')}`;
}

export const log = {
  info: (...a) => console.log(fmt('INFO', a)),
  warn: (...a) => console.warn(fmt('WARN', a)),
  error: (...a) => console.error(fmt('ERROR', a)),
  debug: (...a) => {
    if (process.env.LOG_LEVEL === 'debug') console.log(fmt('DEBUG', a));
  },
};
