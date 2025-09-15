const getDomainFromUrl = (url) => {
    const { hostname } = new URL(url);
    return hostname;
};

const SearchResults = () => {
    return (
        <div>
            <div class="prose lg:prose-xl">
                <p class="leading-7 [&amp;:not(:first-child)]:mt-4 whitespace-pre-wrap break-words">
                    Here are some external resources you may find helpful:
                </p>
            </div>
            <div className="flex flex-col gap-2 p-4 pt-0">
                {props.results.map((result) => (
                    <a
                        key={result.id}
                        href={result.url}
                        target="_blank"
                        className="flex flex-col items-start gap-2 rounded-lg border p-3 text-left text-sm transition-all hover:bg-accent"
                    >
                        <div className="flex w-full flex-col gap-1">
                            <div className="flex items-center">
                                <div className="flex items-center gap-2">
                                    <div className="font-semibold">
                                        {result.title}
                                    </div>
                                </div>
                                <div className="ml-auto text-xs text-muted-foreground">
                                    {getDomainFromUrl(result.url)}
                                </div>
                            </div>
                            <div
                                className="text-xs text-muted-foreground"
                                style={{  // line-clamp-2 class not working for some reason
                                    display: '-webkit-box',
                                    WebkitBoxOrient: 'vertical',
                                    WebkitLineClamp: 2,
                                    overflow: 'hidden',
                                }}
                            >
                                {result.content.substring(0, 300)}
                            </div>
                        </div>
                    </a>
                ))}
            </div>
        </div>
    )
};

export default SearchResults;
