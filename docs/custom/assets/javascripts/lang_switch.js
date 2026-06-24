document.addEventListener('click', function (e) {
    const link = e.target.closest('.md-select__link[hreflang]');
    if (!link) return;

    e.preventDefault();

    const basePrefix = '/zipformer/';
    const langRouteMap = {
        zh: "",
        en: "en/"
    };

    const targetLang = link.getAttribute('hreflang');
    const targetRoutePrefix = langRouteMap[targetLang];
    if (targetRoutePrefix === undefined) return;

    // 剥离基础前缀，拿到站点内部路径
    let rawPath = location.pathname.replace(basePrefix, '');
    // 清除当前路径里的语言前缀（en/）
    Object.values(langRouteMap).forEach(prefix => {
        if (prefix && rawPath.startsWith(prefix)) {
            rawPath = rawPath.slice(prefix.length);
        }
    });

    const finalUrl = (basePrefix + targetRoutePrefix + rawPath).replace(/\/+/g, '/');
    location.href = finalUrl;
});