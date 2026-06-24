document.addEventListener('DOMContentLoaded', function () {
    const basePrefix = '/zipformer/';
    // 语言映射：hreflang -> 路由前缀
    const langRouteMap = {
        zh: "",    // 中文在根目录，无额外前缀
        en: "en/"  // 英文在 en/ 子文件夹
    };
    const langCodes = Object.keys(langRouteMap);

    // 层级限定选择器：仅头部语言下拉内的链接，不会误匹配其他页面下拉
    const langLinks = document.querySelectorAll('.md-header__option .md-select__link');

    langLinks.forEach(link => {
        link.addEventListener('click', function (e) {
            e.preventDefault();
            // 获取目标语言标识 hreflang="zh" / hreflang="en"
            const targetLang = this.getAttribute('hreflang');
            const targetRoutePrefix = langRouteMap[targetLang];

            // 1. 剥离基础前缀，拿到相对站点内部路径
            let rawPath = location.pathname.replace(basePrefix, '');
            // 清除当前路径里所有语言前缀（en/）
            langCodes.forEach(lang => {
                const prefix = langRouteMap[lang];
                if (prefix) rawPath = rawPath.replace(prefix, '');
            });

            // 2. 拼接最终跳转地址
            const targetFullPath = basePrefix + targetRoutePrefix + rawPath;
            // 清洗连续斜杠，防止 /zipformer//en//xxx 这种错误路径
            const finalUrl = targetFullPath.replace(/\/+/g, '/');

            location.href = finalUrl;
        });
    });
});